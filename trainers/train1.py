import csv

import numpy as np
from torch.utils.data import DataLoader

from datasets.MultiviewDataset import ComDataSet
from utils.utils import create_logdir


def load_datasets(hparams):
    if hparams.train_datatype=='Multi-View':
        train_l_dataset = ComDataSet(hparams, mode='train_labeled')
        train_u_dataset = ComDataSet(hparams, mode='train_unlabeled')
        val_dataset = ComDataSet(hparams, mode='val')
        test_dataset = ComDataSet(hparams, mode='test')
        # train_loader = {'l': train_l_loader, 'u': train_u_loader}

        data = [
            np.concatenate([v_l, v_u], axis=0)
            for v_l, v_u in zip(train_l_dataset.mv_data, train_u_dataset.mv_data)
        ]
        targets_l = train_l_dataset.labels.numpy()
        targets_u = train_u_dataset.labels.numpy()
        targets = np.concatenate([targets_l, targets_u], axis=0)
        lb_idx = train_l_dataset.indices
        train_u_dataset.indices = train_u_dataset.indices + len(lb_idx)
        ulb_idx = train_u_dataset.indices
        d_list = train_l_dataset.d_list
        classes_num = train_l_dataset.classes_num

        dataset_dict = {
            'data': data,
            'targets': targets,
            'lb_idx': lb_idx,
            'ulb_idx': ulb_idx,
            'train_lb': train_l_dataset,
            'train_ulb': train_u_dataset,
            'eval': val_dataset,
            'test': test_dataset,
            'dimension_list': d_list,
            'classes_num': classes_num
        }
    else:
        raise Exception('argument dataset must be set to imaging, tabular, multimodal or imaging_and_tabular')
    
    return dataset_dict

def train(hparams):
    #param_dataset_id = hparams.dataset.split(',')
    #datasets = list(map(int, param_dataset_id))

    #for i in range(len(datasets)):
        #hparams.dataset=DATASET[i]
    logdir = create_logdir(hparams)
    hparams.logdir = logdir
    
    '''
    if hparams.train:
        file = open(logdir, 'a')
        file.truncate(0)
        file.write(f"{hparams.dataset}")
        file.write('\n')
        file.close()
    '''

    if hparams.train:
        with open(logdir, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            #writer.writerow([hparams.dataset])

            column_names = [
                'total_loss', 'lb_lre_loss', 'ulb_lre_loss', 'sup_losses', 'fusion_losses',
                'kl_lb_loss', 'kl_ulb_loss', 'acc', 'f1', 'precision', 'recall',
                'auc_ovr', 'auc_ovo', 'gmeans', 'specificity', 'ap',
                'select_ulb_dist', 'lb_select_ulb_dist',
                'Pseudo-label error rate', 'Pseudo-label utilization rate'
            ]
            writer.writerow(column_names)

    dataset_dict = load_datasets(hparams)

    '''
    l_N = len(dataset_dict['train_lb'])
    u_N = len(dataset_dict['train_ulb'])

    target_N = max(l_N, u_N)

    l_indices = np.arange(l_N)
    np.random.shuffle(l_indices)
    u_indices = np.arange(u_N)
    np.random.shuffle(u_indices)

    l_sampler = SubsetRandomSampler(l_indices.tolist() * (target_N // l_N) + l_indices.tolist()[:target_N % l_N])
    u_sampler = SubsetRandomSampler(u_indices.tolist() * (target_N // u_N) + u_indices.tolist()[:target_N % u_N])

    l_loader = DataLoader(
        dataset_dict['train_lb'],
        num_workers=hparams.num_workers,
        batch_size=hparams.l_batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=l_sampler,
        persistent_workers=(hparams.num_workers > 0),
        drop_last=True,
    )

    u_loader = DataLoader(
        dataset_dict['train_ulb'],
        num_workers=hparams.num_workers,
        batch_size=hparams.u_batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=u_sampler,
        persistent_workers=(hparams.num_workers > 0),
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset_dict['eval'],
        num_workers=hparams.num_workers,
        batch_size=hparams.val_batch_size,
        pin_memory=True,
        shuffle=True,
        persistent_workers=(hparams.num_workers > 0),
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset_dict['test'],
        num_workers=hparams.num_workers,
        batch_size=len(dataset_dict['test']),
        pin_memory=True,
        shuffle=True,
        persistent_workers=(hparams.num_workers > 0),
        drop_last=False,
    )

    print(f"Number of train_lb batches: {len(l_loader)}")
    print(f"Number of train_ulb batches: {len(u_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")

    loader_dict = {
        'train_lb': l_loader,
        'train_ulb': u_loader,
        'eval': val_loader,
        'test': test_loader
    }

    '''
    
    l_N, u_N = len(dataset_dict['train_lb']), len(dataset_dict['train_ulb'])

    l_loader = DataLoader(
        dataset_dict['train_lb'], num_workers=hparams.num_workers, batch_size=hparams.l_batch_size, pin_memory=True, shuffle=True, persistent_workers=True, drop_last=True)
    u_loader = DataLoader(dataset_dict['train_ulb'], num_workers=hparams.num_workers, batch_size=10*hparams.l_batch_size, pin_memory=True, shuffle=True, persistent_workers=True, drop_last=True)
    
    val_loader = DataLoader(
        dataset_dict['eval'],
        num_workers=hparams.num_workers, batch_size=len(dataset_dict['eval']) ,
        pin_memory=True, shuffle=False, persistent_workers=True)
    test_loader = DataLoader(
        dataset_dict['test'],
        num_workers=hparams.num_workers, batch_size=len(dataset_dict['test']) ,  
        pin_memory=True, shuffle=False, persistent_workers=True)

    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")

    loader_dict = {}
    loader_dict['train_lb'] = l_loader
    loader_dict['train_ulb'] = u_loader
    loader_dict['eval'] = val_loader
    loader_dict['test'] = test_loader      
    

    if hparams.algorithm_name == 'Multi-View-LTSSL' and hparams.train is True:
        from models.model_vote_right import Model
        model = Model(hparams, dataset_dict, loader_dict)
        model(hparams,column_names)
    elif hparams.algorithm_name == 'Multi-View-LTSSL' and hparams.train is False:
        from models.model_category import Model
        model = Model(hparams, dataset_dict, loader_dict)
        model(hparams)
    else:
        print('Algorithm name not found')

    

    # hparams.repeat_ratio = max(u_N//(hparams.unlabelled_ratio*l_N)-1, 1)
    # print(f'Repeat ratio: {hparams.repeat_ratio}. Load labelled and unlabelled data separately, Labelled data: {len(l_loader)}, Unlabelled data: {len(u_loader)}')
