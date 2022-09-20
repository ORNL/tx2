import logging

import pandas as pd
import torch
from torch import cuda
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

from tx2.wrapper import Wrapper


logger = logging.getLogger()
logger.setLevel(logging.INFO)


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = AutoModel.from_pretrained("bert-base-cased")
        self.l2 = torch.nn.Linear(768, 20)

    def forward(self, ids, mask):
        output_1 = self.l1(ids, mask)
        output = self.l2(output_1[0][:, 0, :])  # use just the [CLS] output embedding
        return output


class EncodedSet(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        print(self.len)

    def __getitem__(self, index):
        text = str(self.data.text[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=True,
        )
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "targets": torch.tensor(self.data.label[index], dtype=torch.long),
        }

    def __len__(self):
        return self.len


def clean_text(text):
    text = str(text)
    # text = text[text.index("\n\n") + 2 :]
    text = text.replace("\n", " ")
    text = text.replace("    ", " ")
    text = text.replace("   ", " ")
    text = text.replace("  ", " ")
    text = text.strip()
    return text


def main():
    # getting newsgroups data from huggingface
    train_data = pd.DataFrame(data=load_dataset("SetFit/20_newsgroups", split="train"))
    test_data = pd.DataFrame(data=load_dataset("SetFit/20_newsgroups", split="test"))

    # setting up pytorch device
    if cuda.is_available():
        device = "cuda"
    elif torch.has_mps:
        device = "mps"
    else:
        device = "cpu"

    model = BERTClass()
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    # clean long white space or extensive character returns
    train_data.text = train_data.text.apply(lambda x: clean_text(x))
    test_data.text = test_data.text.apply(lambda x: clean_text(x))

    # remove empty entries or trivially short ones
    train_cleaned = train_data[train_data["text"].str.len() > 1]
    test_cleaned = test_data[test_data["text"].str.len() > 1]

    train_cleaned.reset_index(drop=True, inplace=True)
    test_cleaned.reset_index(drop=True, inplace=True)

    encodings = (
        train_cleaned[["label", "label_text"]]
        .groupby(["label_text"])
        .apply(lambda x: x["label"].tolist()[0])
        .to_dict()
    )

    wrapper = Wrapper(
        train_texts=train_cleaned.text[:2000],
        train_labels=train_cleaned.label[:2000],
        test_texts=test_cleaned.text[:500],
        test_labels=test_cleaned.label[:500],
        encodings=encodings,
        classifier=model,
        language_model=model.l1,
        tokenizer=tokenizer,
        overwrite=True,
    )
    wrapper.batch_size = 16
    wrapper.prepare()


if __name__ == "__main__":
    main()
