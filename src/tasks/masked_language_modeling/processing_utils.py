import glob
import os
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch

from src import config as cf


def process_data(data_dir, tokenizer, batch_size):
    """ Process data for Masked Language Modeling reading, tokenizing txt files and building a
    dictionary with data corresponding to input ids, attention mask and labels. Also mask 15% of
    the tokens.

    :param data_dir: path to a directory containing the txt files to tokenize
    :type data_dir: str
    :param tokenizer: the tokenizer
    :type tokenizer: BertTokenizer | RobertaTokenizer
    :param batch_size: the batch size
    :type batch_size: int
    :return: a data loader with the training data
    :rtype: torch.utils.data.DataLoader
    """

    files = glob.glob(os.path.join(data_dir, '*.txt'))
    lines = []
    for file in files:
        with open(file, 'r') as f:
            lines.extend(f.read().split('\n'))

    batch = tokenizer(lines, max_length=cf.max_length, padding='max_length', truncation=True)

    labels = torch.tensor([el for el in batch.data['input_ids']])
    mask = torch.tensor([el for el in batch.data['attention_mask']])
    input_ids = labels.detach().clone()

    rand = torch.rand(input_ids.shape)
    mask_arr = (rand < .15) * (input_ids != 0) * (input_ids != 1) * (input_ids != 2)
    for i in range(input_ids.shape[0]):
        selection = torch.flatten(torch.tensor(mask_arr[i].nonzero().tolist(), dtype=torch.long))
        input_ids[i, selection] = 3

    encodings = {'input_ids': input_ids, 'attention_mask': mask, 'labels': labels}

    training_set = MLMDataset(encodings=encodings)

    training_loader = DataLoader(training_set, shuffle=True, batch_size=batch_size)

    return training_loader


class MLMDataset(Dataset):
    """ A Dataset class for a tokenized dataset as a dictionary with "input_ids", "attention_mask" and
    "labels" keys. Allows getting items indexing the data dictionary. To be used with a DataLoader
    for Masked Language Modeling training/testing.
    """

    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return self.encodings['input_ids'].shape[0]

    def __getitem__(self, item):
        return {key: value[item] for key, value in self.encodings.items()}
