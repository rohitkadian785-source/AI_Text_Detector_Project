import torch
from torch.utils.data import Dataset

class AIData(Dataset):
    def __init__(self, texts, labels, processor):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.processor.transform(self.texts[idx])
        label = torch.tensor(self.labels[idx]).float()
        return tokens, label
