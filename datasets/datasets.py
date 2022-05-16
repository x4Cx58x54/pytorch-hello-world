import torch

class PlainImgDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, transforms=None):
        super().__init__()
        self.features = features
        self.labels = labels
        self.transforms = transforms
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        feature = self.features[idx]
        if self.transforms is not None:
            feature = self.transforms(feature)
        label = self.labels[idx]
        return feature, label
