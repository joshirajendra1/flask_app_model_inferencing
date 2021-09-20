from torch.utils.data import Dataset, DataLoader

class InputData(Dataset):
    
    def __init__(self, features, target):
        self.features = features
        self.target   = target
        
    def __getitem__(self, index):
        return self.features[index], self.target[index]
        
    def __len__ (self):
        return len(self.features)

