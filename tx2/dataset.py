from torch.utils.data import Dataset


class EncodedDataset(Dataset):
    def __init__(self, texts, wrapper):
        self.texts = texts
        self.wrapper = wrapper

        self.encoded = self.wrapper.encode(self.texts)

    def __getitem__(self, index):
        if isinstance(self.encoded, dict):
            return {key: self.encoded[key][index] for key in self.encoded.keys()}
        return self.encoded[index]

    def __len__(self):
        return len(self.texts)
