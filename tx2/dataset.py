from torch.utils.data import Dataset

class EncodedDataset(Dataset):
    def __init__(self, texts, wrapper):
        self.texts = texts
        self.wrapper = wrapper

    def __getitem__(self, index):
        encoded = self.wrapper.encode(self.texts[index])
        return encoded

    def __len__(self):
        return len(self.texts)