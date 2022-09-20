from torch.utils.data import Dataset


class EncodedDataset(Dataset):
    def __init__(self, texts, wrapper):
        self.texts = texts
        self.wrapper = wrapper

        self.encoded = self.wrapper.encode(self.texts)

    def __getitem__(self, index):
        return {
            "input_ids": self.encoded["input_ids"][index],
            "attention_mask": self.encoded["attention_mask"][index],
        }

    def __len__(self):
        return len(self.texts)
