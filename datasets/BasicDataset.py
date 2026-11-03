from torch.utils.data import Dataset
import torch

class BasicDataset(Dataset):
    def __init__(self, idx, mv_data, labels, num_classes):
        super().__init__()
        self.idx = idx
        self.mv_data = mv_data
        self.labels = labels
        self.num_classes = num_classes

    def __getitem__(self, index):
        data = [torch.tensor(view[index], dtype=torch.float32) for view in self.mv_data]
        label = torch.tensor(self.labels[index], dtype=torch.long)
        idx = self.idx[index]
        return {'idx_lb': idx, 'x_lb': data, 'y_lb': label}

    def __len__(self):
        return len(self.labels)
