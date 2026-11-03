import csv
import os
from collections import defaultdict
from inspect import signature

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, average_precision_score, classification_report

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
        # for itc, club, and classification
        #self.alpha = self.hparams.alpha
        #self.beta = self.hparams.beta
        #self.gamma = self.hparams.gamma
        #self.rate_uce = self.hparams.rate_uce
        self.update_step = hparams.update_step
        self.memory_step = hparams.memory_step
        self.center = self.nclasses
        self.smoothing = hparams.smoothing
        self.temperature = hparams.temperature
        self.threshold = hparams.threshold
        self.checkpoint = hparams.checkpoint
        self.shadow = {}
        self.backup = {}
        self.ulb_ratio_str = hparams.ulb_ratio
        self.ulb_imb_type_str = hparams.ulb_imb_type
        self.imb_ratio_str = hparams.imb_ratio
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
        self.saved_ulb_idx_set = set() 

        self.adaptive_lb_dest = None
        self.adaptive_lb_dest_loader = None

        lb_class_dist = [0 for _ in range(self.nclasses)]  # labeled samples per class
        for c in self.dataset_dict['train_lb'].labels: 
            lb_class_dist[c] += 1  # count from the original training labels
        lb_class_dist = np.array(lb_class_dist)
        self.lb_dist = torch.from_numpy(lb_class_dist.astype(np.float32)).to(self.gpu)  # class distribution of the labeled training set

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
        self.warm_up = hparams.warmup_epochs
        self.max_epochs = hparams.max_epochs
        self.repeat_ratio = hparams.repeat_ratio
        self.N = hparams.N
        print('Start pseudo label from epoch:', self.warm_up)
        print(f'rate_pseudo: {self.rate_pseudo}')     

        # classification loss
        self.criterion_ce = torch.nn.CrossEntropyLoss()
        self.use_ddp = torch.cuda.device_count() > 1
        print(f'Use DDP: {self.use_ddp}')

    
        self.best_eval_bacc = 0

        # ema
        self.use_ema = hparams.use_ema

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
        self.scheduler = None
        #self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer, warmup_epochs=hparams.warmup_epochs, max_epochs = hparams.max_epochs)
        '''
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, 
            num_training_steps=self.max_epochs,
            num_cycles=7./16.,
            num_warmup_steps=self.warm_up
        )
        '''
        self.logdir = hparams.logdir
        self.bacc = 0
        self.best_epoch = 0
        self.it = 0

    def sharpen_predictions(self, logits, temperature):
        return torch.softmax(logits.detach()/temperature, dim=1)

    def sharpen_predictions1(self, logits, temperature):
        return torch.softmax(logits.detach()/temperature, dim=0)

    def sharpen_predictions2(self, logits, temperature):
        return torch.softmax(logits/temperature, dim=1)

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
            input_args = signature(self.val_step).parameters
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
    
    def re_loss(self, x, re_list, other_recon_list):
        lre1 = 0
        lre2 = 0
        lambda1 = 0.5
        lambda2 = 0.5
        V = len(x)

        for v in range(V):
            # LRE1: self-reconstruction loss of this view
            lre1 = lre1 + F.mse_loss(re_list[v], x[v])
            # LRE2: cross-view reconstruction loss
            other_vs = [u for u in range(V) if u != v]
            for idx_u, u in enumerate(other_vs):
                lre2 = lre2 + F.mse_loss(other_recon_list[v][idx_u], x[u])

        lre = lambda1 * lre1 + lambda2 * lre2

        return lre
    
    def process_log_dict(self, log_dict=None, prefix='train', **kwargs):
        if log_dict is None:
            log_dict = {}

        for arg, var in kwargs.items():
            log_dict[f'{prefix}/' + arg] = var
        return log_dict

    def calculate_metrics(self, y_true, y_pred, y_pred_prob, average='macro', num_classes=None):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average=average)
        precision = precision_score(y_true, y_pred, average=average)
        recall = recall_score(y_true, y_pred, average=average)
        try:
            auc_ovr = roc_auc_score(y_true, y_pred_prob, multi_class='ovr', average=None)
        except Exception:
            auc_ovr = [None] * num_classes
        try:
            auc_ovo = roc_auc_score(y_true, y_pred_prob, multi_class='ovo', average=None)
        except Exception:
            auc_ovo = [None] * num_classes
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
            ap = average_precision_score(y_true, y_pred_prob, average=None)
        except Exception:
            ap = [None] * num_classes
        try:
            ap1 = average_precision_score(y_true, y_pred_prob, average=average)
        except Exception:
            ap1 = None

        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        specificity_list = []
        sensitivity_list = []
        gmeans_list = []
        acc_list = []
        for i in range(num_classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            acc_i = TP / (TP + FN) if (TP + FN) > 0 else np.nan
            specificity_list.append(specificity)
            sensitivity_list.append(sensitivity)
            gmeans_list.append(np.sqrt(sensitivity * specificity))
            acc_list.append(acc_i)

        specificity_mean = np.mean(specificity_list)
        recall_mean = np.mean(sensitivity_list)
        gmeans = np.sqrt(recall_mean * specificity_mean)

        report = classification_report(
            y_true, y_pred, labels=list(range(num_classes)),
            output_dict=True, zero_division=0
        )

        per_class = {}
        for i in range(num_classes):
            str_i = str(i)
            per_class[str_i] = {
                'acc': acc_list[i],
                'f1': report[str_i]['f1-score'],
                'precision': report[str_i]['precision'],
                'recall': report[str_i]['recall'],
                'auc_ovr': auc_ovr[i] if auc_ovr is not None else None,
                'auc_ovo': auc_ovo[i] if auc_ovo is not None else None,
                'gmeans': gmeans_list[i],
                'specificity': specificity_list[i],
                'ap': ap[i] if ap is not None else None,
                'support': report[str_i]['support'],
            }

        return {
            'acc': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'auc_ovr': auc1,
            'auc_ovo': auc2,
            'gmeans': gmeans,
            'specificity': specificity_mean,
            'ap': ap1,
            'per_class': per_class
        }


    def forward(self, hparams):

        ulb_ratio_str = f"{self.ulb_ratio_str:.2f}"
        ulb_imb_type_str = self.ulb_imb_type_str
        imb_ratio_str = f"{self.imb_ratio_str:.1f}"
        lr = f"{self.lr}"
        save_name = f"model_best.pth"
        filename = self.dataset + '_' + ulb_ratio_str + '_' + ulb_imb_type_str + '_' + imb_ratio_str + '_' + lr + '_' + save_name
        save_filename = os.path.join(self.checkpoint, self.dataset, filename)
        self.test_step(save_filename)


    def val_step(self, idx_lb, x_lb, y_lb, prefix='val'):
        share_lb_feature, all_result = self.model(x_lb, self.lb_select_ulb_dist, 'val')

        metric_names = ['acc', 'f1', 'precision', 'recall', 'auc_ovr', 'auc_ovo', 'gmeans', 'specificity', 'ap']
        all_metrics = []
        y_true = y_lb.cpu().numpy()
        per_class_metrics = {}

        '''
        for v_idx, v_logits in enumerate(share_lb_feature):
            y_pred_prob = torch.softmax(v_logits, dim=-1).cpu().numpy()
            y_pred = np.argmax(y_pred_prob, axis=-1)
            metrics = self.calculate_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_pred_prob=np.array(y_pred_prob),
                average='macro',
                num_classes=v_logits.shape[1]
            )
            all_metrics.append(metrics)
        '''

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

        # macro-averaged metrics
        all_keys = all_metrics[0].keys()
        avg_metrics = {}
        for k in all_keys:
            if k == 'per_class':
                continue
            values = [m[k] if m[k] is not None else np.nan for m in all_metrics]
            avg_metrics[k] = np.nanmean(values)

        # per-class metrics (averaged over views)
        if 'per_class' in all_metrics[0]:
            class_ids = list(all_metrics[0]['per_class'].keys())
            class_metric_names = list(all_metrics[0]['per_class'][class_ids[0]].keys())
            per_class_avg = {}
            for class_idx in class_ids:
                for metric_name in class_metric_names:
                    vals = [
                        m['per_class'][class_idx][metric_name] if (
                            m['per_class'] is not None and 
                            class_idx in m['per_class'] and 
                            m['per_class'][class_idx][metric_name] is not None
                        ) else np.nan
                        for m in all_metrics
                    ]
                    per_class_avg[f'class{class_idx}/{metric_name}'] = np.nanmean(vals)

        else:
            per_class_avg = {}

        # only these two groups are written to log_dict
        log_dict = {}
        for name in metric_names:
            log_dict[f'{prefix}/avg/{name}'] = avg_metrics[name]
        for k, v in per_class_avg.items():
            log_dict[f'{prefix}/avg/{k}'] = v

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
        all_log_dicts = []
        per_class_buf = defaultdict(list)
        with torch.no_grad():
            for data_test in self.loader_dict['test']:  # the test split loader
                log_dict, avg_metrics = self.val_step(**self.process_batch(**data_test), prefix='test')
                all_avg_metrics.append(avg_metrics)
                all_log_dicts.append(log_dict)
                # collect per-class values
                for k, v in log_dict.items():
                    if '/class' in k and v is not None:
                        per_class_buf[k].append(v)

        final_metrics = {}
        if all_avg_metrics:
            metric_keys = all_avg_metrics[0].keys()
            for k in metric_keys:
                values = [m[k] for m in all_avg_metrics if m[k] is not None]
                final_metrics[k] = float(np.nanmean(values))

        # per-class values of the last batch are written out
        per_class_mean = {k: float(np.nanmean(vs)) for k, vs in per_class_buf.items()}

        '''
        with open(self.logdir, 'a') as file:
            file.write('---------------test---------------\n')
            file.write(str(final_metrics))
            file.write('\n')
            file.write('Per-class metrics:\n')
            for k in log_dict:
                if '/class' in k:
                    file.write(f'{k}: {log_dict[k]}\n')
        '''
    

        row_dict = {**final_metrics, **per_class_mean}

        # append metric names and values to the CSV file
        with open(self.logdir, mode='a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # first row: metric names
            header_row = list(row_dict.keys())
            writer.writerow(header_row)

            # second row: metric values
            value_row = [row_dict[k] for k in header_row]
            writer.writerow(value_row)

        print("\n====== Test Metrics ======")
        for key, value in final_metrics.items():
            print(f"{key}: {value:.4f}")





