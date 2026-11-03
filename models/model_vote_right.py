import csv
import os
from collections import Counter, defaultdict
from inspect import signature

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, average_precision_score
from torch.utils.data import DataLoader

from datasets.BasicDataset import BasicDataset
from models.extract import FeatureExtract


class Model(nn.Module):
    def __init__(self, hparams, dataset_dict, loader_dict):
        super().__init__()

        self.gpu = hparams.device
        self.dataset_dict = dataset_dict
        self.loader_dict = loader_dict
        self.nclasses = self.dataset_dict['classes_num']
        self.model = FeatureExtract(n_input=self.dataset_dict['dimension_list'], n_z=hparams.projection_dim, nLabel=self.nclasses).to(self.gpu)
        self.lr = hparams.lr

        self.update_step = hparams.update_step
        self.memory_step = hparams.memory_step
        self.smoothing = hparams.smoothing
        self.temperature = hparams.temperature
        self.threshold = hparams.threshold
        self.checkpoint = hparams.checkpoint
        self.shadow = {}
        self.backup = {}
        self.ulb_ratio_str = hparams.ulb_ratio  # ratio of labeled data
        self.ulb_imb_type_str = hparams.ulb_imb_type
        self.imb_ratio_str = hparams.imb_ratio  # imbalance ratio (largest class / smallest class)
        self.train = hparams.train
        self.test = hparams.test
        
        self.data = self.dataset_dict['data']
        self.targets = self.dataset_dict['targets']
        self.lb_idx =  self.dataset_dict['lb_idx']
        self.lb_idx_stable = self.dataset_dict['lb_idx']
        self.ulb_idx =  self.dataset_dict['ulb_idx']
        self.dataset = hparams.dataset

        self.current_x = []
        self.current_y = self.targets[self.lb_idx_stable]
        self.current_idx = None
        self.current_one_hot_y = None

        self.select_ulb_idx = None
        self.select_ulb_label = None
        self.select_ulb_pseudo_label = None
        self.select_ulb_pseudo_label_distribution = None
        self.ulb_vote_buffer = defaultdict(list)
        self.ulb_gt_buffer = {}  

        self.adaptive_lb_dest = None
        self.adaptive_lb_dest_loader = None

        lb_class_dist = [0 for _ in range(self.nclasses)]  # labeled samples per class
        for c in self.dataset_dict['train_lb'].labels: 
            lb_class_dist[c] += 1  # count from the original training labels
        lb_class_dist = np.array(lb_class_dist)
        self.lb_dist = torch.from_numpy(lb_class_dist.astype(np.float32)).to(self.gpu)  # class distribution of the labeled training set
        self.lb_dist_original = self.lb_dist.clone()

        ulb_class_dist = [0 for _ in range(self.nclasses)]  # unlabeled samples per class
        for c in self.dataset_dict['train_ulb'].labels:
            ulb_class_dist[c] += 1  # count from the unlabeled split
        ulb_class_dist = np.array(ulb_class_dist)
        self.ulb_dist = torch.from_numpy(ulb_class_dist.astype(np.float32)).to(self.gpu)  # class distribution of the unlabeled training split

        self.select_ulb_dist = torch.zeros(self.nclasses).to(self.gpu)  # pseudo-labeled samples per class
        lb_ulb_class_dist = lb_class_dist + ulb_class_dist  # class distribution of the whole training pool
        self.lb_ulb_dist = torch.from_numpy(lb_ulb_class_dist.astype(np.float32)).to(self.gpu)  # class distribution of the whole training pool
        self.lb_select_ulb_dist = self.lb_dist + self.select_ulb_dist  # labeled + pseudo-labeled samples per class

        self.T = hparams.temperature
        self.rate_pseudo = hparams.rate_pseudo
        self.start_epoch = hparams.start_epoch   # start epoch using pseudo label
        self.warm_up = hparams.warmup_epochs  # warm-up epochs without unlabeled data
        self.max_epochs = hparams.max_epochs
        #self.repeat_ratio = hparams.repeat_ratio
        self.N = hparams.N  # length of the training schedule
        print('Start pseudo label from epoch:', self.warm_up)
        print(f'rate_pseudo: {self.rate_pseudo}')     

        # classification loss
        self.use_ddp = torch.cuda.device_count() > 1
        print(f'Use DDP: {self.use_ddp}')

        # ema
        self.use_ema = hparams.use_ema
        if self.use_ema:
            from .ema import ModelEMA

            self.ema_model = ModelEMA(hparams, self.model, 0.999)
            self.teacher = self.ema_model.ema 

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
        self.scheduler = None

        self.logdir = hparams.logdir
        self.f1 = 0
        self.best_epoch = 0
        self.it = 0
        self.ulb_seen_in_window = 0
        self.pseudo_selected_in_window = 0 

        self.batch_size = hparams.l_batch_size
        self.temperature_f = 0.5
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask
    
    def contrastive_loss_lb(self, h_i, h_j):
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def contrastive_loss_ulb(self, h_i, h_j):
        N = 2 * 10 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)

        sim = torch.matmul(h, h.T) / self.temperature_f
        sim_i_j = torch.diag(sim, 10 * self.batch_size)
        sim_j_i = torch.diag(sim, -(10 * self.batch_size))

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def sharpen_predictions(self, logits, temperature):
        return torch.softmax(logits.detach()/temperature, dim=1)

    def sharpen_predictions1(self, logits, temperature):
        return torch.softmax(logits.detach()/temperature, dim=0)

    def sharpen_predictions2(self, logits, temperature):
        return torch.softmax(logits, dim=1)
    
    @torch.no_grad()
    def teacher_forward(self, x, dist, mode='train'):
        if not self.use_ema:
            return self.model(x, dist, mode)
        self.teacher.eval()
        return self.teacher(x, dist, mode)

    def compute_prototypes_softlabel(self, features_list, y_lb, nclasses):
        prototypes_list = []
        for features in features_list:
            class_sum = y_lb.t().float() @ features       # [nclasses, feature_dim]
            class_count = y_lb.sum(dim=0).unsqueeze(1)  # [nclasses, 1]
            prototype = class_sum / (class_count + 1e-8)
            prototypes_list.append(prototype)
        return prototypes_list

    def process_batch(self, input_args=None, **kwargs):
        if input_args is None:
            input_args = signature(self.train_step).parameters
            input_args = list(input_args.keys())

        input_dict = {}

        for arg, var in kwargs.items():
            if not arg in input_args:
                continue

            if var is None:
                continue

            if isinstance(var, dict):
                var = {k: v.cuda(self.gpu) for k, v in var.items()}
            elif isinstance(var, list):
                var = [v.cuda(self.gpu) for v in var]
            else:
                var = var.cuda(self.gpu)
            input_dict[arg] = var
        return input_dict
    
    def ce_loss(self, logits, targets, reduction='none'):
        if logits.shape == targets.shape:
            # one-hot target
            log_pred = F.log_softmax(logits, dim=-1)
            nll_loss = torch.sum(-targets * log_pred, dim=1)
            if reduction == 'none':
                return nll_loss
            else:
                return nll_loss.mean()
        else:
            log_pred = F.log_softmax(logits, dim=-1)
            return F.nll_loss(log_pred, targets, reduction=reduction)
    
    def re_loss(self, x, re_list):
        lre1 = 0
        lre2 = 0
        V = len(x)

        for v in range(V):
            # LRE1: self-reconstruction loss of this view
            lre1 = lre1 + F.mse_loss(re_list[v], x[v])

        lre = lre1

        return lre
    
    def process_log_dict(self, log_dict=None, prefix='train', **kwargs):
        if log_dict is None:
            log_dict = {}

        for arg, var in kwargs.items():
            log_dict[f'{prefix}/' + arg] = var
        return log_dict

    def get_save_dict(self):
        if self.use_ema:
            ema_to_save = (
                self.ema_model.ema.module
                if hasattr(self.ema_model.ema, "module")
                else self.ema_model.ema
            )
            save_dict = {
                'model': self.model.state_dict(),
                'ema_model': ema_to_save.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'it': self.it + 1,
                'epoch': self.epoch + 1
            }
        else:
            save_dict = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'it': self.it + 1,
                'epoch': self.epoch + 1
            }
        return save_dict

    def save_model(self, save_name, save_path):
        save_path = os.path.join(save_path, self.dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        ulb_ratio_str = f"{self.ulb_ratio_str:.2f}"
        ulb_imb_type_str = self.ulb_imb_type_str
        imb_ratio_str = f"{self.imb_ratio_str:.1f}"
        lr = f"{self.lr}"
        save_name = f"model_best.pth"
        filename = self.dataset + '_' + ulb_ratio_str + '_' + ulb_imb_type_str + '_' + imb_ratio_str + '_' + lr + '_' + save_name
        save_filename = os.path.join(self.checkpoint, self.dataset, filename)
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_filename)
        print(f"model saved: {save_filename}")

    def calculate_metrics(self, y_true, y_pred, y_pred_prob, average='macro', num_classes=None):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average=average)
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        try:
            auc1 = roc_auc_score(y_true, y_pred_prob, multi_class='ovr')
        except Exception as e:
            #print(e)
            auc1 = None
        try:
            auc2 = roc_auc_score(y_true, y_pred_prob, multi_class='ovo')
        except Exception as e:
            auc2 = None
        try:
            ap = average_precision_score(y_true, y_pred_prob, average=average)
        except Exception:
            ap = None
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        specificity_list = []
        sensitivity_list = []
        for i in range(num_classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity_list.append(specificity)
            sensitivity_list.append(sensitivity)
        specificity_mean = np.mean(specificity_list)
        recall_mean = np.mean(sensitivity_list)
        gmeans = np.sqrt(recall_mean * specificity_mean)
        return {'acc':acc, 'f1': f1, 'precision': precision, 'recall': recall, 'auc_ovr': auc1, 'auc_ovo': auc2, 
                'gmeans': gmeans, 'specificity': specificity_mean, 'ap': ap}

    def forward(self, hparams, column_names):
        if self.train:
            self.model.train()
            self.ulb_seen_in_window = 0
            self.pseudo_selected_in_window = 0
            for epoch in range(self.start_epoch, self.max_epochs):
                self.epoch = epoch
                
                # warm-up stage: unlabeled data are not used
                if self.epoch <= self.warm_up:
                    self.adaptive_lb_dest_loader = self.loader_dict['train_lb']
                    self.lb_select_ulb_dist = self.lb_dist
                    self.select_ulb_dist = torch.ones(self.nclasses).to(self.gpu)

                else:
                    if self.epoch % self.memory_step == 0:
                        self.current_x = None
                        self.current_y = None
                        self.current_idx = None
                        self.current_one_hot_y = None
                        self.select_ulb_pseudo_label_distribution = None

                        select_ulb_idx_to_label = {}  # ground-truth labels seen for each unlabeled index
                        select_ulb_idx_to_pseudo_label = {}  # pseudo-labels predicted for each index

                        for ulb_idx, ulb_pseudo_label, ulb_label in zip(self.select_ulb_idx, self.select_ulb_pseudo_label, self.select_ulb_label):
                            if ulb_idx.item() in select_ulb_idx_to_label:
                                select_ulb_idx_to_label[ulb_idx.item()].append(ulb_label.item())
                            else:
                                select_ulb_idx_to_label[ulb_idx.item()] = [ulb_label.item()]

                            if ulb_idx.item() in select_ulb_idx_to_pseudo_label:
                                select_ulb_idx_to_pseudo_label[ulb_idx.item()].append(ulb_pseudo_label.item())
                            else:
                                select_ulb_idx_to_pseudo_label[ulb_idx.item()] = [ulb_pseudo_label.item()]

                        select_ulb_unique_idx = torch.unique(self.select_ulb_idx)

                        select_ulb_unique_label = []
                        select_ulb_unique_pseudo_label = []
                        select_ulb_unique_pseudo_label_distribution = []

                        for ulb_unique_idx in select_ulb_unique_idx:
                            ulb_unique_label = select_ulb_idx_to_label[ulb_unique_idx.item()]
                            ulb_unique_pseudo_label = select_ulb_idx_to_pseudo_label[ulb_unique_idx.item()]

                            ulb_unique_pseudo_label_distribution = torch.zeros(self.nclasses)
                            for item in ulb_unique_pseudo_label:
                                ulb_unique_pseudo_label_distribution[item] += 1.0
                            ulb_unique_pseudo_label_distribution = ulb_unique_pseudo_label_distribution / torch.sum(ulb_unique_pseudo_label_distribution)

                            # process the ground-truth label                                                        
                            select_ulb_unique_label.append(torch.tensor([ulb_unique_label[0]]))

                            # process the pseudo-label
                            if len(ulb_unique_pseudo_label) > 2:
                                most_common_label = Counter(ulb_unique_pseudo_label).most_common(1)[0][0]
                                most_common_number = Counter(ulb_unique_pseudo_label).most_common(1)[0][1]
                                select_ulb_unique_pseudo_label.append(torch.tensor([most_common_label]))
                            else:
                                select_ulb_unique_pseudo_label.append(torch.tensor([-1]))
                            # process the pseudo-label distribution
                            select_ulb_unique_pseudo_label_distribution.append(ulb_unique_pseudo_label_distribution.unsqueeze(0))

                        if select_ulb_unique_label:
                            select_ulb_unique_label = torch.cat(select_ulb_unique_label)
                            select_ulb_unique_pseudo_label = torch.cat(select_ulb_unique_pseudo_label)
                            select_ulb_unique_pseudo_label_distribution = torch.cat(select_ulb_unique_pseudo_label_distribution)
                        else:
                            device = self.gpu
                            select_ulb_unique_label = torch.empty(0, dtype=torch.long, device=device)
                            select_ulb_unique_pseudo_label = torch.empty(0, dtype=torch.long, device=device)
                            select_ulb_unique_pseudo_label_distribution = torch.empty(0, self.nclasses, device=device)

                        # keep every tensor on the same device
                        device = select_ulb_unique_idx.device

                        if select_ulb_unique_idx.numel() > 0:

                            valid_mask = select_ulb_unique_pseudo_label != -1

                            self.select_ulb_idx = select_ulb_unique_idx[valid_mask]
                            self.select_ulb_label = select_ulb_unique_label[valid_mask]
                            self.select_ulb_pseudo_label = select_ulb_unique_pseudo_label[valid_mask]
                            self.select_ulb_pseudo_label_distribution = \
                                select_ulb_unique_pseudo_label_distribution[valid_mask]

                        else:
                            # no pseudo-labeled sample at all
                            self.select_ulb_idx = torch.empty(0, dtype=torch.long, device=device)
                            self.select_ulb_label = torch.empty(0, dtype=torch.long, device=device)
                            self.select_ulb_pseudo_label = torch.empty(0, dtype=torch.long, device=device)
                            self.select_ulb_pseudo_label_distribution = \
                                torch.empty(0, self.nclasses, device=device)



                        self.select_ulb_dist = torch.zeros(self.nclasses).cuda(self.gpu)
                        for item in self.select_ulb_pseudo_label:
                            self.select_ulb_dist[int(item)] += 1

                        self.lb_select_ulb_dist = self.lb_dist + self.select_ulb_dist

                        print('select_ulb_dist:\n' + np.array_str(np.array(self.select_ulb_dist.cpu())))
                        print('lb_select_ulb_dist:\n' + np.array_str(np.array(self.lb_select_ulb_dist.cpu())))

                        self.current_idx = np.concatenate((self.lb_idx, self.select_ulb_idx.cpu().numpy()), axis=0)
                        if isinstance(self.data, list):
                            self.current_x = [view[self.current_idx] for view in self.data]
                        elif isinstance(self.data, np.ndarray):
                            self.current_x = self.data[self.current_idx]
                        elif isinstance(self.data, torch.Tensor):
                            self.current_x = self.data[self.current_idx].cpu().numpy()
                        else:
                            raise TypeError(f"Unsupported data type: {type(self.data)}")


                        if self.select_ulb_pseudo_label.numel() > 0:
                            pseudo_np = self.select_ulb_pseudo_label
                        else:
                            pseudo_np = np.array([], dtype=self.targets.dtype)

                        self.current_y = np.concatenate((self.targets[self.lb_idx], pseudo_np), axis=0)

                        # reset select ulb idx and its pseudo label
                        self.select_ulb_idx = None
                        self.select_ulb_label = None
                        self.select_ulb_pseudo_label = None
                        self.select_ulb_pseudo_label_distribution = None

                    # update step on the labeled data
                    if (self.epoch - self.warm_up - self.memory_step) % self.update_step == 0:
                        print(str(self.epoch) + ': Update the labeled data.')

                        self.adaptive_lb_dest = BasicDataset(self.current_idx, self.current_x, self.current_y, self.nclasses)
                        
                        self.adaptive_lb_dest_loader = DataLoader(
                            self.adaptive_lb_dest,
                            num_workers=hparams.num_workers,
                            batch_size=hparams.l_batch_size,
                            pin_memory=True,
                            shuffle=True,
                            persistent_workers=(hparams.num_workers > 0),
                            drop_last=True,
                        )

                        self.current_x = None
                        self.current_y = None
                        self.current_idx = None
            
                total_loss = 0
                total_num = 0
                total_selected = 0
                total_ulb = 0
                total_error = 0

                for data_lb, data_ulb in zip(self.adaptive_lb_dest_loader, 
                                            self.loader_dict['train_ulb']):
                    if self.epoch < self.warm_up:
                        self.log_dict, loss = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                    else:
                        self.log_dict, loss, selected, all_ulb, error_count = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                        total_selected += selected
                        total_ulb += all_ulb
                        total_error += error_count

                        print("total_selected", total_selected, "total_ulb:", total_ulb)

                    self.optimizer.zero_grad()
                    print("total_loss", loss, "grad_fn:", loss.grad_fn)
                    with torch.autograd.detect_anomaly():
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    if self.use_ema:
                        self.ema_model.update(self.model)

                    total_loss += loss.item() * data_lb['x_lb'][0].shape[0]  # all views share the same batch size
                    total_num += data_lb['x_lb'][0].shape[0]

                util_ratio = total_selected / total_ulb if total_selected > 0 else 0
                error_ratio = total_error / total_selected if total_selected > 0 else 0
                    
                avg_loss = total_loss / total_num

                print("validating...")

                if self.use_ema:
                    self.test_model = self.ema_model.ema
                self.test_model.eval()

                all_avg_metrics = []

                with torch.no_grad():
                    for data_val in self.loader_dict['eval']:
                        _, avg_metrics = self.val_step(**self.process_batch(**data_val))
                        all_avg_metrics.append(avg_metrics)

                final_metrics = {}
                if all_avg_metrics:
                    metric_keys = all_avg_metrics[0].keys()
                    for k in metric_keys:
                        values = [m[k] for m in all_avg_metrics if m[k] is not None]
                        final_metrics[k] = float(np.nanmean(values))
                
                with open(self.logdir, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=column_names)

                    row_data = {
                        'total_loss': avg_loss,
                        'lb_lre_loss': self.log_dict.get('lb_lre_loss', 0),
                        'ulb_lre_loss': self.log_dict.get('ulb_lre_loss', 0),
                        'sup_losses': self.log_dict.get('sup_losses', 0),
                        'fusion_losses': self.log_dict.get('fusion_losses', 0),
                        'kl_lb_loss': self.log_dict.get('kl_lb_loss', 0),
                        'kl_ulb_loss': self.log_dict.get('kl_ulb_loss', 0),
                        'acc': final_metrics.get('acc', 0),
                        'f1': final_metrics.get('f1', 0),
                        'precision': final_metrics.get('precision', 0),
                        'recall': final_metrics.get('recall', 0),
                        'auc_ovr': final_metrics.get('auc_ovr', 0),
                        'auc_ovo': final_metrics.get('auc_ovo', 0),
                        'gmeans': final_metrics.get('gmeans', 0),
                        'specificity': final_metrics.get('specificity', 0),
                        'ap': final_metrics.get('ap', 0),
                        'select_ulb_dist': '',
                        'lb_select_ulb_dist': '',
                        'Pseudo-label error rate': error_ratio,
                        'Pseudo-label utilization rate': util_ratio
                    }
                    writer.writerow(row_data)


                filename = f"model_best.pth"
                if final_metrics['f1'] > self.f1:
                    self.save_model(filename, self.checkpoint)
                    self.f1= final_metrics['f1']
                    self.best_epoch = self.epoch

                self.it += 1

            with open(self.logdir, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([self.best_epoch])
                writer.writerow([self.f1])
        
        if self.test:
            ulb_ratio_str = f"{self.ulb_ratio_str:.2f}"
            ulb_imb_type_str = self.ulb_imb_type_str
            imb_ratio_str = f"{self.imb_ratio_str:.1f}"
            lr = f"{self.lr}"
            save_name = f"model_best.pth"
            filename = self.dataset + '_' + ulb_ratio_str + '_' + ulb_imb_type_str + '_' + imb_ratio_str + '_' + lr + '_' + save_name
            save_filename = os.path.join(self.checkpoint, self.dataset, filename)
            self.test_step(save_filename)

    def train_step(self, idx_lb, x_lb, y_lb, idx_ulb, x_ulb, y_ulb):
        num_lb = y_lb.shape[0]
        share_lb_feature, share_lb_classification, share_lb_fusion, re_lb_list, cfs_lb = self.model(x_lb, self.lb_select_ulb_dist, 'train')
        share_ulb_feature, share_ulb_classification, share_ulb_fusion, re_ulb_list, cfs_ulb = self.model(x_ulb, self.lb_select_ulb_dist, 'train')
        share_ulb_feature_t, share_ulb_classification_t, share_ulb_fusion_t, _, _ = self.teacher_forward(x_ulb, self.lb_select_ulb_dist, 'train')
        
        if self.epoch < self.warm_up:
            # smoothed label tensor (one-hot + label smoothing)
            lb_smooth = torch.zeros(num_lb, self.nclasses).to(self.gpu)
            # start from the smoothing value everywhere
            lb_smooth.fill_(self.smoothing/ (self.nclasses - 1))
            # true class gets 1 - smoothing, the others get smoothing / (C - 1)
            lb_smooth.scatter_(1, y_lb.unsqueeze(1), 1.0 - self.smoothing)
            # replace the original labels with the smoothed ones
            y_lb = lb_smooth

            # reconstruction loss
            lb_lre_loss = self.re_loss(x_lb, re_lb_list)
            
            # cross-entropy on share_lb_classification against the smoothed y_lb; 'mean' averages over the batch
            # logits_x_lb_w gets a log-prior correction that counters the long-tailed training distribution
            lb_bias = torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist))  # shape [n_class]

            sup_losses = 0
            for logits in share_lb_classification:
                logits_bias = logits + lb_bias
                loss = self.ce_loss(logits_bias, y_lb, reduction='mean')
                sup_losses += loss
            #sup_losses = sup_losses / len(share_lb_classification)  # averaging over views can also be omitted

            # fusion loss
            fusion_losses = 0
            fusion_losses = self.ce_loss(share_lb_fusion + lb_bias, y_lb, reduction='mean')

            # KL divergence
            num_views = len(share_lb_classification)
            batch_size = share_lb_classification[0].shape[0]
            nclasses = self.nclasses

            pred_lb_softmax = []
            for v in range(num_views):
                logits = share_lb_classification[v]  # [batch, nclasses]
                #prob = torch.softmax(logits, dim=1)   # [batch, nclasses]
                prob = self.sharpen_predictions2(logits, self.temperature)
                pred_lb_softmax.append(prob)

            fusion_logits = share_lb_fusion  # [batch, nclasses]   
            #pred_lb_share_softmax = torch.softmax(fusion_logits, dim=1) # [batch, nclasses]
            pred_lb_share_softmax = self.sharpen_predictions2(fusion_logits, self.temperature)

            kl_loss = 0
            for v in range(len(pred_lb_softmax)):
                kl = F.kl_div(
                    (pred_lb_softmax[v] + 1e-8).log(),
                    pred_lb_share_softmax,
                    reduction='mean'
                )
                kl_loss += kl

            loss_list = []
            for v in range(num_views):
                for w in range(v + 1, num_views):
                    loss_list.append(self.contrastive_loss_lb(cfs_lb[v], cfs_lb[w]))

            loss_contrastive = sum(loss_list)
            total_loss = lb_lre_loss + 10 * sup_losses + 10 * fusion_losses + kl_loss + 0.5 * loss_contrastive
            
            log_dict = self.process_log_dict(total_loss=total_loss,
                                            lb_lre_loss=lb_lre_loss,
                                            ulb_lre_loss=0,
                                            sup_losses=10 * sup_losses,
                                            fusion_losses=10 * fusion_losses,
                                            kl_lb_loss=kl_loss,
                                            kl_ulb_loss=0)
        
            return log_dict, total_loss
        

        else:
            # smoothed label tensor (one-hot + label smoothing)
            lb_smooth = torch.zeros(num_lb, self.nclasses).to(self.gpu)
            # start from the smoothing value everywhere
            lb_smooth.fill_(self.smoothing/ (self.nclasses - 1))
            # true class gets 1 - smoothing, the others get smoothing / (C - 1)
            lb_smooth.scatter_(1, y_lb.unsqueeze(1), 1.0 - self.smoothing)
            # replace the original labels with the smoothed ones
            y_lb = lb_smooth

            # y_lb_int = torch.argmax(y_lb, dim=1)

            prototypes_list = self.compute_prototypes_softlabel(share_lb_feature, y_lb, self.nclasses)  # per-view prototypes, each of shape [nclasses, feature_dim]

            pred_prototypes_ulb = []
            for v in range(len(share_ulb_feature_t)):
                feat = share_ulb_feature_t[v]            # [batch, feature_dim]
                protos = prototypes_list[v]            # [nclasses, feature_dim]
                
                diff = feat.unsqueeze(1) - protos.unsqueeze(0)    # [batch, nclasses, feature_dim]
                dist2 = (diff ** 2).sum(dim=2)                   # [batch, nclasses]
                
                # turn distances into probabilities with softmax
                dist2 = -dist2
                prob = self.sharpen_predictions(dist2, self.temperature)  # [batch, nclasses]
                
                # predicted class of every sample (largest probability)
                pred_label = torch.argmax(prob, dim=1)  # [batch]
                
                pred_prototypes_ulb.append(pred_label)

            pred_ulb_softmax = []
            pred_ulb_class = []

            pred_ulb_share_softmax = []
            pred_ulb_share_class = []
            
            for logits in share_ulb_classification_t:
                prob = self.sharpen_predictions(logits, self.temperature) # [batch, nclasses]
                pred = torch.argmax(prob, dim=1)           # [batch]
                pred_ulb_softmax.append(prob)
                pred_ulb_class.append(pred)

            for logits in share_ulb_fusion_t:
                #prob = self.sharpen_predictions1(logits, self.temperature) # [batch, nclasses]
                prob = self.sharpen_predictions1(logits, self.temperature)
                pred = torch.argmax(prob, dim=0)           # [batch]
                pred_ulb_share_softmax.append(prob)
                pred_ulb_share_class.append(pred)

            batch_size = pred_prototypes_ulb[0].shape[0]
            n_views = len(pred_prototypes_ulb)
            
            mask = torch.zeros(batch_size, dtype=torch.bool, device=self.gpu)
            majority_labels = torch.full((batch_size,), -1, dtype=torch.long, device=self.gpu)

            for i in range(batch_size):
                # Step 1: keep the sample only if more than half of the views agree, and keep only the agreeing views
                class_labels = [pred_ulb_class[v][i].item() for v in range(n_views)]
                label_counter = Counter(class_labels)
                majority_class, most_common_count = label_counter.most_common(1)[0]

                if most_common_count <= n_views / 2:
                    continue  # no majority agreement, drop the sample

                # keep only the views that predict the majority class
                consistent_view_indices = [v for v in range(n_views) if class_labels[v] == majority_class]

                # Step 2: compare each kept view with its prototype and keep the views whose prediction matches
                logits_list = []
                for v in consistent_view_indices:
                    proto_label = pred_prototypes_ulb[v][i].item()
                    if proto_label != majority_class:
                        continue  # this view disagrees with its prototype
                    logits_list.append(share_ulb_classification_t[v][i])  # [nclasses]

                if len(logits_list) == 0:
                    continue  # no view left after the prototype check, drop the sample

                # Step 3: keep the pseudo-label only if all remaining views are confident
                all_above_threshold = True
                for logits in logits_list:
                    probs = self.sharpen_predictions1(logits, self.temperature)
                    max_prob = torch.max(probs).item()
                    if max_prob <= self.threshold * (1 - self.smoothing):
                        all_above_threshold = False
                        break

                if all_above_threshold:
                    mask[i] = True
                    majority_labels[i] = majority_class


            if self.select_ulb_idx is not None and self.select_ulb_pseudo_label is not None and self.select_ulb_label is not None:
                self.select_ulb_idx = torch.cat([self.select_ulb_idx, idx_ulb[mask]], dim=0)
                self.select_ulb_label = torch.cat([self.select_ulb_label, y_ulb[mask]], dim=0)
                self.select_ulb_pseudo_label = torch.cat([self.select_ulb_pseudo_label, majority_labels[mask]], dim=0)
            else:
                self.select_ulb_idx = idx_ulb[mask]
                self.select_ulb_label = y_ulb[mask]
                self.select_ulb_pseudo_label = majority_labels[mask]

            # reconstruction loss
            lb_lre_loss = self.re_loss(x_lb, re_lb_list)
            ulb_lre_loss = self.re_loss(x_ulb, re_ulb_list)

            # cross-entropy loss
            lb_bias = torch.log(self.lb_select_ulb_dist / torch.sum(self.lb_select_ulb_dist))  # shape [n_class]

            sup_losses = 0
            for logits in share_lb_classification:
                logits_bias = logits + lb_bias
                loss = self.ce_loss(logits_bias, y_lb, reduction='mean')
                sup_losses += loss

            #sup_losses = sup_losses / len(share_lb_classification)  # averaging over views can also be omitted

            # fusion loss
            fusion_losses = 0
            fusion_losses = self.ce_loss(share_lb_fusion + lb_bias, y_lb, reduction='mean')

            # KL divergence
            num_views = len(share_lb_classification)
            nclasses = self.nclasses

            pred_lb_softmax = []
            for v in range(num_views):
                logits = share_lb_classification[v]  # [batch, nclasses]
                #prob = torch.softmax(logits, dim=1)
                prob = self.sharpen_predictions2(logits, self.temperature)
                pred_lb_softmax.append(prob)

            fusion_logits = share_lb_fusion  # [batch, nclasses]

            pred_lb_share_softmax = self.sharpen_predictions2(fusion_logits, self.temperature)

            kl_lb_loss = 0
            for v in range(len(pred_lb_softmax)):
                kl = F.kl_div(
                    (pred_lb_softmax[v] + 1e-8).log(),
                    pred_lb_share_softmax,
                    reduction='mean'
                )
                kl_lb_loss += kl

            
            pred_ulb_softmax = []
            for v in range(num_views):
                logits = share_ulb_classification[v]  # [batch, nclasses]
                prob = self.sharpen_predictions2(logits, self.temperature)
                pred_ulb_softmax.append(prob)

            fusion_logits = share_ulb_fusion  # [batch, nclasses]
            pred_ulb_share_softmax = self.sharpen_predictions2(fusion_logits, self.temperature)

            kl_ulb_loss = 0
            for v in range(len(pred_ulb_softmax)):
                kl = F.kl_div(
                    (pred_ulb_softmax[v] + 1e-8).log(),
                    pred_ulb_share_softmax,
                    reduction='mean'
                )
                kl_ulb_loss += kl

            #logits_all_bias = all_result + lb_bias
            #loss_pre = self.ce_loss(logits_all_bias, y_lb, reduction='mean')


            with torch.no_grad():
                t_prob = self.sharpen_predictions2(share_ulb_fusion_t, self.temperature)

            s_prob = self.sharpen_predictions2(share_ulb_fusion, self.temperature)

            temporal_kl = F.kl_div((s_prob + 1e-8).log(), t_prob, reduction='mean')


            loss_list1 = []
            loss_list2 = []
            for v in range(num_views):
                for w in range(v + 1, num_views):
                    loss_list1.append(self.contrastive_loss_lb(cfs_lb[v], cfs_lb[w]))
                    loss_list2.append(self.contrastive_loss_ulb(cfs_ulb[v], cfs_ulb[w]))

            loss_contrastive1 = sum(loss_list1)
            loss_contrastive2 = sum(loss_list2)

            # supervised loss for the other stages
            total_loss = lb_lre_loss + ulb_lre_loss + 10 * sup_losses + 10 * fusion_losses + kl_lb_loss + kl_ulb_loss + temporal_kl + 0.5 * loss_contrastive1 + 0.5 * loss_contrastive2
            
            log_dict = self.process_log_dict(total_loss=total_loss,
                                            lb_lre_loss=lb_lre_loss,
                                            ulb_lre_loss=ulb_lre_loss,
                                            sup_losses=10 * sup_losses,
                                            fusion_losses=10 * fusion_losses,
                                            kl_lb_loss=kl_lb_loss,
                                            kl_ulb_loss=kl_ulb_loss)
        
            selected = mask.sum().item()
            total_ulb = mask.shape[0]
            error_count = (majority_labels[mask] != y_ulb[mask]).sum().item()

            return log_dict, total_loss, selected, total_ulb, error_count


    def val_step(self, idx_lb, x_lb, y_lb, prefix='val'):
        share_lb_feature, all_result = self.test_model(x_lb, self.lb_select_ulb_dist, 'val')

        metric_names = ['acc', 'f1', 'precision', 'recall', 'auc_ovr', 'auc_ovo', 'gmeans', 'specificity', 'ap']
        all_metrics = []
        y_true = y_lb.cpu().numpy()

        y_pred_prob = torch.softmax(all_result, dim=-1).cpu().numpy()
        y_pred = np.argmax(y_pred_prob, axis=-1)
        metrics = self.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_pred_prob=np.array(y_pred_prob),
            average='macro',
            num_classes=all_result.shape[1]
        )
        all_metrics.append(metrics)
        

        all_keys = all_metrics[0].keys()
        avg_metrics = {}
        for k in all_keys:
            values = [m[k] if m[k] is not None else np.nan for m in all_metrics]
            avg_metrics[k] = np.nanmean(values)

        log_dict = {}

        for v_idx, metrics in enumerate(all_metrics):
            for name in metric_names:
                log_dict[f'{prefix}/view{v_idx}/{name}'] = metrics[name]

        for name in metric_names:
            log_dict[f'{prefix}/avg/{name}'] = avg_metrics[name]

        log_dict = self.process_log_dict(log_dict, prefix=prefix)

        return log_dict, avg_metrics

    def test_step(self, checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.gpu)

        if self.use_ema and 'ema_model' in checkpoint:
            self.model.load_state_dict(checkpoint['ema_model'])
            print("Loaded EMA model.")
        else:
            self.model.load_state_dict(checkpoint['model'])
            print("Loaded standard model.")

        self.model.eval()

        all_avg_metrics = []
        with torch.no_grad():
            for data_test in self.loader_dict['test']:  # the test split loader
                self.log_dict, avg_metrics = self.val_step(**self.process_batch(**data_test), prefix='test')
                all_avg_metrics.append(avg_metrics)

        final_metrics = {}
        if all_avg_metrics:
            metric_keys = all_avg_metrics[0].keys()
            for k in metric_keys:
                values = [m[k] for m in all_avg_metrics if m[k] is not None]
                final_metrics[k] = float(np.nanmean(values))
        
        file = open(self.logdir, 'a')
        file.write('---------------test---------------')
        file.write('\n')
        file.write(str(final_metrics))
        file.write('\n')
        file.close()


        print("\n====== Test Metrics ======")
        for key, value in final_metrics.items():
            print(f"{key}: {value:.4f}")
