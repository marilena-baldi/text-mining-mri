import glob
import json
import os.path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from src import config as cf


def process_data(dataset_dir, tokenizer, batch_size):
    """ Process data for Named Entity Recognition reading a csv dataset, splitting the data,
    instantiating NERDataset objects and returning data loaders. Also saves the datasets (both the
    full version and those grouped by users) so they can be loaded if they already exist.

    :param dataset_dir: path to the folder with the source csv data files with columns as
    (Sentence #,Word,Tag)
    :type dataset_dir: str
    :param tokenizer: the tokenizer
    :type tokenizer: BertTokenizer | RobertaTokenizer
    :param batch_size: the batch size
    :type batch_size: int
    :return: a training loader, a testing loader and a dictionary with labels to ids mapping
    :rtype: (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, dict[str, str])
    """

    df_paths = glob.glob(os.path.join(dataset_dir, '*.csv'))
    if not (os.path.exists(cf.ner_train_path) and os.path.exists(cf.ner_valid_path)
            and os.path.exists(cf.ner_test_path)):

        labels_to_ids = {}
        train_sets, val_sets, test_sets = [], [], []
        for df_path in df_paths:
            data = pd.read_csv(df_path)
            data = data.fillna(method='ffill')

            user_id = data["UserId"].unique()
            labels_to_ids.update({k: v for v, k in enumerate(data.Tag.unique())})

            train_dataset, valid_dataset, test_dataset = split_data(dataframe=data)
            train_dataset = group_data(dataframe=train_dataset)
            valid_dataset = group_data(dataframe=valid_dataset)
            test_dataset = group_data(dataframe=test_dataset)

            train_sets.append(train_dataset)
            val_sets.append(valid_dataset)
            test_sets.append(test_dataset)

            train_path = cf.ner_train_path.replace('.csv', '_' + str(int(user_id[0])) + '.csv')
            train_dataset.to_csv(train_path, index=False)
            valid_path = cf.ner_valid_path.replace('.csv', '_' + str(int(user_id[0])) + '.csv')
            valid_dataset.to_csv(valid_path, index=False)
            test_path = cf.ner_test_path.replace('.csv', '_' + str(int(user_id[0])) + '.csv')
            test_dataset.to_csv(test_path, index=False)

        train_dataset = pd.concat(train_sets)
        valid_dataset = pd.concat(val_sets)
        test_dataset = pd.concat(test_sets)

        train_dataset.to_csv(cf.ner_train_path, index=False)
        valid_dataset.to_csv(cf.ner_valid_path, index=False)
        test_dataset.to_csv(cf.ner_test_path, index=False)

        with open(cf.ner_l2id_path, 'w+') as labels_file:
            json.dump(labels_to_ids, labels_file)

    else:
        train_dataset = pd.read_csv(cf.ner_train_path)
        valid_dataset = pd.read_csv(cf.ner_valid_path)
        test_dataset = pd.read_csv(cf.ner_test_path)

        with open(cf.ner_l2id_path, 'r') as labels_file:
            labels_to_ids = json.load(labels_file)

    train_dataset = train_dataset.sample(frac=1, random_state=1).reset_index(drop=True)
    valid_dataset = valid_dataset.sample(frac=1, random_state=1).reset_index(drop=True)
    test_dataset = test_dataset.sample(frac=1, random_state=1).reset_index(drop=True)

    training_set = NERDataset(dataframe=train_dataset, tokenizer=tokenizer,
                              max_len=cf.max_length, labels_to_ids=labels_to_ids)
    validation_set = NERDataset(dataframe=valid_dataset, tokenizer=tokenizer,
                                max_len=cf.max_length, labels_to_ids=labels_to_ids)
    testing_set = NERDataset(dataframe=test_dataset, tokenizer=tokenizer,
                             max_len=cf.max_length, labels_to_ids=labels_to_ids)

    training_loader = DataLoader(dataset=training_set, shuffle=True, batch_size=batch_size)
    validation_loader = DataLoader(dataset=validation_set, shuffle=True, batch_size=batch_size)
    testing_loader = DataLoader(dataset=testing_set, shuffle=True, batch_size=batch_size)

    return training_loader, validation_loader, testing_loader, labels_to_ids


def split_data(dataframe):
    """ Split a dataframe in training, validation and testing set

    :param dataframe: the input dataframe
    :type dataframe: pd.Dataframe
    :return: the training, validation and testing sets
    :rtype: (pd.Dataframe, pd.Dataframe, pd.Dataframe)
    """

    report_ids = dataframe["ReportId"].unique()

    np.random.seed(1)
    np.random.shuffle(report_ids)

    train_idx_end = int(len(report_ids) * cf.ner_train_size)
    val_idx_end = int(len(report_ids) * (cf.ner_train_size + cf.ner_valid_size))

    train_df = dataframe[dataframe["ReportId"].isin(report_ids[:train_idx_end])]
    val_df = dataframe[dataframe["ReportId"].isin(report_ids[train_idx_end:val_idx_end])]
    test_df = dataframe[dataframe["ReportId"].isin(report_ids[val_idx_end:])]

    return train_df, val_df, test_df


def group_data(dataframe):
    """ Group the input data basing on the amount of reports

    :param dataframe: the input data
    :type dataframe: pd.Dataframe
    :return: the grouped data
    :rtype: pd.Dataframe
    """

    grouped_dataframe = dataframe.copy()
    grouped_dataframe['sentence'] = grouped_dataframe[
        ["UserId", "ReportId", 'Sentence #', 'Word', 'Tag']
    ].groupby(['Sentence #'])['Word'].transform(lambda x: ' '.join(x))
    grouped_dataframe['word_labels'] = grouped_dataframe[
        ["UserId", "ReportId", 'Sentence #', 'Word', 'Tag']
    ].groupby(['Sentence #'])['Tag'].transform(lambda x: ','.join(x))

    grouped_dataframe = grouped_dataframe[
        ["UserId", "ReportId", 'Sentence #', 'sentence', 'word_labels']
    ].drop_duplicates().reset_index(drop=True)

    return grouped_dataframe


def tokenize_and_preserve_labels(in_sentence, in_labels, tokenizer):
    """ Tokenize a sentence mapping the corresponding labels for every sub-word.

    :param in_sentence: the input sentence to tokenize
    :type in_sentence: str
    :param in_labels: the input sentence labels
    :type in_labels: str
    :param tokenizer: the tokenizer
    :type tokenizer: BertTokenizer | RobertaTokenizer
    :return: the tokenized sentence and the corresponding labels
    :rtype: (List[str], List[str])
    """

    tokenized_sentence = []
    labels = []

    sentence = in_sentence.strip()

    for word, label in zip(sentence.split(), in_labels.split(",")):
        tokenized_word = tokenizer.tokenize(word)
        num_sub_words = len(tokenized_word)

        tokenized_sentence.extend(tokenized_word)
        labels.extend([label] * num_sub_words)

    return tokenized_sentence, labels


class NERDataset(Dataset):
    """ A Dataset class for a dataset as a dataframe with columns as (sentence, word_labels). Allows
    getting items indexing the dataframe. Tokenizes and pads the sentences returning input_ids,
    attention_mask and labels. To be used with a DataLoader for Named Entity Recognition
    training/validation/testing.
    """

    def __init__(self, dataframe, tokenizer, max_len, labels_to_ids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids

    def __getitem__(self, index):
        sentence = self.data.sentence[index]
        word_labels = self.data.word_labels[index]
        tokenized_sentence, labels = tokenize_and_preserve_labels(sentence,
                                                                  word_labels,
                                                                  self.tokenizer)

        tokenized_sentence = [self.tokenizer.special_tokens_map['cls_token']] + tokenized_sentence + \
                             [self.tokenizer.special_tokens_map['sep_token']]

        labels.insert(0, 'O')
        labels.insert(-1, 'O')

        if len(tokenized_sentence) > self.max_len:
            tokenized_sentence = tokenized_sentence[:self.max_len]
            labels = labels[:self.max_len]

        else:
            tokenized_sentence = tokenized_sentence + [
                self.tokenizer.special_tokens_map['pad_token']
                for _ in range(self.max_len - len(tokenized_sentence))
            ]
            labels = labels + ['O' for _ in range(self.max_len - len(labels))]

        attention_mask = [1 if tok != self.tokenizer.special_tokens_map['pad_token']
                          else 0 for tok in tokenized_sentence]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        label_ids = [self.labels_to_ids[label] for label in labels]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }

    def __len__(self):
        return self.len
