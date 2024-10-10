import glob
import json
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from src import config as cf


def balance_by_cls(dataset):
    # select data UserId,ReportId,Sentence,Class
    imbalance_threshold = cf.imbalance_threshold
    key_columns = ['UserId','ReportId','Sentence #','Cls']
    sentences_classes = dataset.groupby(key_columns).head(1)[key_columns]
    colname = 'Cls'
    clss = [e[1] for e in enumerate(sentences_classes[getattr(sentences_classes, colname).notna()][colname].unique())]
    cls_count = {}
    for k in clss:
        cls_count[k] = (sentences_classes[colname]==k).sum()

    min_size = min(cls_count.values())
    max_cls_size = min_size + round((min_size*imbalance_threshold)/100.0)
    cls_size = {key: max_cls_size if cls_count[key] > max_cls_size else cls_count[key] for key, value in
                cls_count.items()}

    samples_x_cls= []

    for k in  cls_size:
        sample_per_cls = sentences_classes[sentences_classes['Cls']==k].sample(cls_size[k])
        samples_x_cls.append(sample_per_cls)

    samples = pd.concat(samples_x_cls)
    samples = samples.sample(frac=1)
    # rebuild
    sampled = []
    #dataset_sample = pd.DataFrame(columns=dataset.columns)
    for index, row in samples.iterrows():
        ds = dataset[(dataset['UserId']==row['UserId']) &
                     (dataset['ReportId']==row['ReportId']) &
                     (dataset['Sentence #']==row['Sentence #']) &
                     (dataset['Cls']==row['Cls'])]
        sampled.append(ds)

    dataset_sample = pd.concat(sampled)

    dataset_sample.reset_index(drop=True, inplace=True)
    return dataset_sample


def process_data(dataset_dir, tokenizer, batch_size):
    """ Process data for SequenceClassification reading a csv dataset, splitting the data,
    instantiating SeqDataset objects and returning data loaders. Also saves the datasets (both the
    full version and those grouped by users) so they can be loaded if they already exist.

    :param dataset_dir: path to the folder with the source csv data files with columns as
    (Sentence #,Word,Cls)
    :type dataset_dir: str
    :param tokenizer: the tokenizer
    :type tokenizer: BertTokenizer | RobertaTokenizer
    :param batch_size: the batch size
    :type batch_size: int
    :return: a training loader, a testing loader and a dictionary with labels to ids mapping
    :rtype: (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, dict[str, str])
    """

    df_paths = glob.glob(os.path.join(dataset_dir, '*.csv'))

    if not (os.path.exists(cf.seq_train_path) and os.path.exists(cf.seq_valid_path)
            and os.path.exists(cf.seq_test_path)):

        labels_to_ids = {}
        dataset = None
        for df_path in df_paths:
            data = pd.read_csv(df_path)
            data = data.fillna(method='ffill')
            if (dataset is None):
                dataset = data
            else:
                dataset=dataset.append(data)

        labels_to_ids.update({k: v for v, k in enumerate(dataset[dataset.Cls.ne('')].Cls.unique())})

        dataset = balance_by_cls(dataset)

        train_dataset, valid_dataset, test_dataset = split_data(dataframe=dataset)

        train_dataset = group_data(dataframe=train_dataset)
        valid_dataset = group_data(dataframe=valid_dataset)
        test_dataset = group_data(dataframe=test_dataset)
        # clean row without class
        train_dataset = train_dataset.drop(train_dataset[train_dataset['sentence_labels'] == ''].index)

        train_dataset.to_csv(cf.seq_train_path, index=False)
        valid_dataset.to_csv(cf.seq_valid_path, index=False)
        test_dataset.to_csv(cf.seq_test_path, index=False)
        #labels_to_ids.pop('')
        with open(cf.seq_l2id_path, 'w+') as labels_file:
            json.dump(labels_to_ids, labels_file)

    else:
        train_dataset = pd.read_csv(cf.seq_train_path,keep_default_na=False)
        valid_dataset = pd.read_csv(cf.seq_valid_path,keep_default_na=False)
        test_dataset = pd.read_csv(cf.seq_test_path,keep_default_na=False)

        with open(cf.seq_l2id_path, 'r') as labels_file:
            labels_to_ids = json.load(labels_file)

    train_dataset = train_dataset.sample(frac=1, random_state=1).reset_index(drop=True)
    valid_dataset = valid_dataset.sample(frac=1, random_state=1).reset_index(drop=True)
    test_dataset = test_dataset.sample(frac=1, random_state=1).reset_index(drop=True)

    training_set = SeqDataset(dataframe=train_dataset, tokenizer=tokenizer,
                              max_len=cf.max_length, labels_to_ids=labels_to_ids)
    validation_set = SeqDataset(dataframe=valid_dataset, tokenizer=tokenizer,
                                max_len=cf.max_length, labels_to_ids=labels_to_ids)
    testing_set = SeqDataset(dataframe=test_dataset, tokenizer=tokenizer,
                             max_len=cf.max_length, labels_to_ids=labels_to_ids)

    training_loader = DataLoader(dataset=training_set, shuffle=True, batch_size=batch_size)
    validation_loader = DataLoader(dataset=validation_set, shuffle=True, batch_size=batch_size)
    testing_loader = DataLoader(dataset=testing_set, shuffle=True, batch_size=batch_size)

    return training_loader, validation_loader, testing_loader, labels_to_ids



def process_data_user(dataset_dir, tokenizer, batch_size):
    """ Process data for SequenceClassification reading a csv dataset, splitting the data,
    instantiating SeqDataset objects and returning data loaders. Also saves the datasets (both the
    full version and those grouped by users) so they can be loaded if they already exist.

    :param dataset_dir: path to the folder with the source csv data files with columns as
    (Sentence #,Word,Cls)
    :type dataset_dir: str
    :param tokenizer: the tokenizer
    :type tokenizer: BertTokenizer | RobertaTokenizer
    :param batch_size: the batch size
    :type batch_size: int
    :return: a training loader, a testing loader and a dictionary with labels to ids mapping
    :rtype: (torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader, dict[str, str])
    """

    df_paths = glob.glob(os.path.join(dataset_dir, '*.csv'))

    if not (os.path.exists(cf.seq_train_path) and os.path.exists(cf.seq_valid_path)
            and os.path.exists(cf.seq_test_path)):

        labels_to_ids = {}
        train_sets, val_sets, test_sets = [], [], []
        for df_path in df_paths:
            data = pd.read_csv(df_path)
            data = data.fillna(method='ffill')

            user_id = data["UserId"].unique()
            labels_to_ids.update({k: v for v, k in enumerate(data.Cls.unique())})

            train_dataset, valid_dataset, test_dataset = split_data(dataframe=data)

            train_dataset = group_data(dataframe=train_dataset)
            valid_dataset = group_data(dataframe=valid_dataset)
            test_dataset = group_data(dataframe=test_dataset)

            train_sets.append(train_dataset)
            val_sets.append(valid_dataset)
            test_sets.append(test_dataset)

            train_path = cf.seq_train_path.replace('.csv', '_' + str(int(user_id[0])) + '.csv')
            train_dataset.to_csv(train_path, index=False)
            valid_path = cf.seq_valid_path.replace('.csv', '_' + str(int(user_id[0])) + '.csv')
            valid_dataset.to_csv(valid_path, index=False)
            test_path = cf.seq_test_path.replace('.csv', '_' + str(int(user_id[0])) + '.csv')
            test_dataset.to_csv(test_path, index=False)

        train_dataset = pd.concat(train_sets)
        valid_dataset = pd.concat(val_sets)
        test_dataset = pd.concat(test_sets)

        train_dataset.to_csv(cf.seq_train_path, index=False)
        valid_dataset.to_csv(cf.seq_valid_path, index=False)
        test_dataset.to_csv(cf.seq_test_path, index=False)
        with open(cf.seq_l2id_path, 'w+') as labels_file:
            json.dump(labels_to_ids, labels_file)

    else:
        train_dataset = pd.read_csv(cf.seq_train_path)
        valid_dataset = pd.read_csv(cf.seq_valid_path)
        test_dataset = pd.read_csv(cf.seq_test_path)

        with open(cf.seq_l2id_path, 'r') as labels_file:
            labels_to_ids = json.load(labels_file)

    train_dataset = train_dataset.sample(frac=1, random_state=1).reset_index(drop=True)
    valid_dataset = valid_dataset.sample(frac=1, random_state=1).reset_index(drop=True)
    test_dataset = test_dataset.sample(frac=1, random_state=1).reset_index(drop=True)

    training_set = SeqDataset(dataframe=train_dataset, tokenizer=tokenizer,
                              max_len=cf.max_length, labels_to_ids=labels_to_ids)
    validation_set = SeqDataset(dataframe=valid_dataset, tokenizer=tokenizer,
                                max_len=cf.max_length, labels_to_ids=labels_to_ids)
    testing_set = SeqDataset(dataframe=test_dataset, tokenizer=tokenizer,
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

    np.random.seed(26)
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
                            ["UserId", "ReportId", 'Sentence #', 'Word', 'Cls']
                            ].groupby(['Sentence #'])['Word'].transform(lambda x: ' '.join(x))


    grouped_dataframe['sentence_labels'] = grouped_dataframe[
                            ["UserId", "ReportId", 'Sentence #', 'Word', 'Cls']
                            ].groupby(['Sentence #'])['Cls'].transform(lambda x: x)


    grouped_dataframe = grouped_dataframe[
                            ["UserId", "ReportId", 'Sentence #', "sentence", "sentence_labels"]
                        ].drop_duplicates().reset_index(drop=True)

    return grouped_dataframe


def tokenize(sentence, tokenizer):
    """ Tokenize the words of a sentence

    :param sentence: the input sentence to tokenize
    :type sentence: str
    :param tokenizer: the tokenizer
    :type tokenizer: BertTokenizer | RobertaTokenizer
    :return: the tokenized sentence
    :rtype: List[str]
    """

    tokenized_sentence = []
    for word in sentence.strip().split():
        tokenized_word = tokenizer.tokenize(word)
        tokenized_sentence.extend(tokenized_word)

    return tokenized_sentence


class SeqDataset(Dataset):
    """ A Dataset class for a dataset as a dataframe with columns as (sentence, sentence_label). Allows
    getting items indexing the dataframe. Tokenizes and pads sentences returning input_ids,
    attention_mask and labels. To be used with a DataLoader for Sequence Classification
    training/testing.
    """

    def __init__(self, dataframe, tokenizer, max_len, labels_to_ids):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.labels_to_ids = labels_to_ids

    def __getitem__(self, index):
        sentence = self.data.sentence[index]
        label = self.data.sentence_labels[index]
        tokenized_sentence = tokenize(sentence=sentence, tokenizer=self.tokenizer)

        tokenized_sentence = [
                                self.tokenizer.special_tokens_map['cls_token']] + \
                                tokenized_sentence + \
                                [self.tokenizer.special_tokens_map['sep_token']
                            ]

        if len(tokenized_sentence) > self.max_len:
            tokenized_sentence = tokenized_sentence[:self.max_len]

        else:
            tokenized_sentence = tokenized_sentence + [
                                    self.tokenizer.special_tokens_map['pad_token']
                                    for _ in range(self.max_len - len(tokenized_sentence))
                                ]

        attention_mask = [1 if tok != self.tokenizer.special_tokens_map['pad_token']
                          else 0 for tok in tokenized_sentence]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized_sentence)
        if (type(label) == float and math.isnan(label)):
            label = 'NaN'
        if not label in self.labels_to_ids:
            print('!')
        label_id = self.labels_to_ids[label]

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }

    def __len__(self):
        return self.len
