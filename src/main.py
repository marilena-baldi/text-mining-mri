import torch
import os
from torch.utils.data import DataLoader, ConcatDataset

import config as cf
import tasks.named_entity_recognition.train_test_utils as ner_training
import tasks.named_entity_recognition.processing_utils as ner_processing
import tasks.masked_language_modeling.train_test_utils as mlm_training
import tasks.masked_language_modeling.processing_utils as mlm_processing
import tasks.tokenization.train_utils as tok_training
import tasks.sequence_classification.train_test_utils as seq_training
import tasks.sequence_classification.processing_utils as seq_processing


def get_tokenizer():
    """ Get the tokenizer according to the configuration parameters.

    :return: the tokenizer
    :rtype: BertTokenizer | RobertaTokenizer
    """

    tok_weights = cf.tok_weights_dir

    if cf.model_config == cf.bert_config:
        if cf.model_config['tok_train']:
            tok_training.train_wp(data_dir=cf.tok_data_dir, model_save_path=cf.tok_weights_dir)

        else:
            tok_weights = cf.bert_pretrained_weights_ita

    else:
        tok_training.train_bl(data_dir=cf.tok_data_dir, model_save_path=cf.tok_weights_dir)

    tokenizer = cf.model_config['tok_model'].from_pretrained(tok_weights, max_len=cf.max_length)

    return tokenizer


def train_mlm(tokenizer, num_epochs, batch_size, learning_rate):
    """
    Train a model for Masked Language Modeling using a tokenizer and save the weights

    :param tokenizer: the tokenizer object
    :type tokenizer: BertTokenizer | RobertaTokenizer
    :param num_epochs: the number of epochs, defaults to cf.mlm_epochs
    :type num_epochs: int, optional
    :param batch_size: the batch size
    :type batch_size: int
    :param learning_rate: the learning rate
    :type learning_rate: float
    """

    training_loader = mlm_processing.process_data(data_dir=cf.tok_data_dir,
                                                  tokenizer=tokenizer,
                                                  batch_size=batch_size)

    config = cf.model_config['conf_model'](vocab_size=cf.vocab_size,
                                           max_position_embeddings=514,
                                           hidden_size=768,
                                           num_attention_heads=12,
                                           num_hidden_layers=6,
                                           type_vocab_size=2)
    model = cf.model_config['mlm_model'](config=config)
    model.to(cf.device)
    model.resize_token_embeddings(len(tokenizer))

    experiment_name = f"{cf.model_config['name']}_e_{num_epochs}_bs_{batch_size}_lr_{learning_rate}"
    mlm_weights_dir = os.path.join(cf.mlm_weights_dir, experiment_name)
    mlm_training.run_training(model=model, data_loader=training_loader, num_epochs=num_epochs,
                              learning_rate=learning_rate, model_save_dir=mlm_weights_dir)


def get_mlm_model(tokenizer, num_epochs=cf.mlm_epochs, learning_rate=cf.mlm_learning_rate,
                  batch_size=cf.mlm_batch_size):
    """ Return the MLM basing on the configuration setup

    :param tokenizer: the tokenizer
    :type tokenizer: BertTokenizer | RobertaTokenizer
    :param num_epochs: the number of epochs, defaults to cf.mlm_epochs
    :type num_epochs: int, optional
    :param learning_rate: the learning rate, defaults to cf.mlm_learning_rate
    :type learning_rate: float, optional
    :param batch_size: the batch size, defaults to cf.mlm_batch_size
    :type batch_size: int, optional
    :return: the path of the model weights
    :rtype: str
    """

    experiment_name = f"{cf.model_config['name']}_e_{num_epochs}_bs_{batch_size}_lr_{learning_rate}"
    weights = os.path.join(cf.mlm_weights_dir, experiment_name)

    if cf.model_config == cf.bert_config:
        if cf.model_config['mlm_train']:
            train_mlm(tokenizer=tokenizer,
                      num_epochs=num_epochs,
                      learning_rate=learning_rate,
                      batch_size=batch_size)

        else:
            weights = cf.bert_pretrained_weights_ita

    else:
        train_mlm(tokenizer=tokenizer,
                  num_epochs=num_epochs,
                  learning_rate=learning_rate,
                  batch_size=batch_size)

    return weights


def train_ner(training_loader, labels_to_ids, weights, learning_rate, num_epochs, batch_size):
    """ Do training for Named Entity Recognition

    :param training_loader: the training loader
    :type training_loader: torch.utils.data.DataLoader
    :param labels_to_ids: the labels and the corresponding ids
    :type labels_to_ids: dict[str, str]
    :param weights: the path to the BERT weights for which to do transfer learning
    :type weights: str
    :param learning_rate: the learning rate
    :type learning_rate: float
    :param num_epochs: the number of epochs
    :type num_epochs: int
    :param batch_size: the batch_size
    :type batch_size: int
    """

    model = cf.model_config['ner_model'].from_pretrained(weights,
                                                         num_labels=len(labels_to_ids),
                                                         return_dict=False)
    model.to(cf.device)

    experiment_name = f"{cf.model_config['name']}_e_{num_epochs}_bs_{batch_size}_lr_{learning_rate}"
    ner_weights_dir = os.path.join(cf.ner_weights_dir, experiment_name)
    ner_training.run_training(model=model,
                              data_loader=training_loader,
                              num_epochs=num_epochs,
                              learning_rate=learning_rate,
                              model_save_dir=ner_weights_dir)


def test_ner(testing_loader, labels_to_ids, learning_rate, num_epochs, batch_size):
    """ Do testinf for Named Entity Recognition

    :param testing_loader: the testing loader
    :type testing_loader: torch.utils.data.DataLoader
    :param labels_to_ids: the labels and the corresponding ids
    :type labels_to_ids: dict[str, str]
    :param learning_rate: the learning rate
    :type learning_rate: float
    :param num_epochs: the number of epochs
    :type num_epochs: int
    :param batch_size: the batch_size
    :type batch_size: int
    """

    experiment_name = f"{cf.model_config['name']}_e_{num_epochs}_bs_{batch_size}_lr_{learning_rate}"
    ner_weights_dir = os.path.join(cf.ner_weights_dir, experiment_name)
    model = cf.model_config['ner_model'].from_pretrained(ner_weights_dir,
                                                         num_labels=len(labels_to_ids),
                                                         return_dict=False)

    model.to(cf.device)

    ner_training.run_testing(model=model,
                             data_loader=testing_loader,
                             labels_to_ids=labels_to_ids,
                             model_dir=ner_weights_dir)


def train_val_ner():
    """ Do validation for Named Entity Recognition, saving the results """

    tokenizer = get_tokenizer()

    ner_weights = get_mlm_model(tokenizer=tokenizer)

    epochs_nums = cf.ner_hyperparameters["epochs"]
    batch_sizes = cf.ner_hyperparameters["batch_sizes"]
    learning_rates = cf.ner_hyperparameters["learning_rates"]

    for batch_size in batch_sizes:
        training_loader, validation_loader, _, labels_to_ids = ner_processing.process_data(dataset_dir=cf.ner_data_dir,
                                                                                           tokenizer=tokenizer,
                                                                                           batch_size=batch_size)
        for epochs_num in epochs_nums:
            for learning_rate in learning_rates:
                train_ner(training_loader=training_loader,
                          labels_to_ids=labels_to_ids,
                          weights=ner_weights,
                          num_epochs=epochs_num,
                          learning_rate=learning_rate,
                          batch_size=batch_size)
                test_ner(testing_loader=validation_loader,
                         labels_to_ids=labels_to_ids,
                         learning_rate=learning_rate,
                         num_epochs=epochs_num,
                         batch_size=batch_size)


def train_test_ner(epochs=cf.ner_epochs, learning_rate=cf.ner_learning_rate,
                   batch_size=cf.ner_batch_size):
    """ Do training and testing for Named Entity Recognition

    :param epochs: the number of epochs, defaults to cf.ner_epochs
    :type epochs: int, optionsl
    :param learning_rate: the learning rate, defaults to cf.ner_learning_rate
    :type learning_rate: float, optional
    :param batch_size: the batch size, defaults to cf.ner_batch_size
    :type batch_size: int, optional
    """

    tokenizer = get_tokenizer()

    ner_weights = get_mlm_model(tokenizer=tokenizer)

    training_loader, validation_loader, testing_loader, labels_to_ids = ner_processing.process_data(
        dataset_dir=cf.ner_data_dir,
        tokenizer=tokenizer,
        batch_size=batch_size)

    train_val_loader = DataLoader(
        ConcatDataset([training_loader.dataset, validation_loader.dataset]),
        shuffle=True,
        batch_size=training_loader.batch_size)

    train_ner(training_loader=train_val_loader,
              labels_to_ids=labels_to_ids,
              weights=ner_weights,
              num_epochs=epochs,
              learning_rate=learning_rate,
              batch_size=batch_size)
    test_ner(testing_loader=testing_loader,
             labels_to_ids=labels_to_ids,
             learning_rate=learning_rate,
             num_epochs=epochs,
             batch_size=batch_size)


def train_seq(training_loader, labels_to_ids, weights, learning_rate, num_epochs, batch_size):
    """ Do training for Sequence Classification

    :param training_loader: the training loader
    :type training_loader: torch.utils.data.DataLoader
    :param labels_to_ids: the labels and the corresponding ids
    :type labels_to_ids: dict[str, str]
    :param weights: the path to the BERT weights for which to do transfer learning
    :type weights: str
    :param learning_rate: the learning rate
    :type learning_rate: float
    :param num_epochs: the number of epochs
    :type num_epochs: int
    :param batch_size: the batch_size
    :type batch_size: int
    """

    model = cf.model_config['seq_model'].from_pretrained(weights,
                                                         num_labels=len(labels_to_ids),
                                                         return_dict=False)
    model.to(cf.device)

    experiment_name = f"{cf.model_config['name']}_e_{num_epochs}_bs_{batch_size}_lr_{learning_rate}"
    seq_weights_dir = os.path.join(cf.seq_weights_dir, experiment_name)
    seq_training.run_training(model=model,
                              data_loader=training_loader,
                              num_epochs=num_epochs,
                              learning_rate=learning_rate,
                              model_save_dir=seq_weights_dir)


def test_seq(testing_loader, labels_to_ids, learning_rate, num_epochs, batch_size):
    """ Do testing for Sequence Classification

    :param testing_loader: the testing loader
    :type testing_loader: torch.utils.data.DataLoader
    :param labels_to_ids: the labels and the corresponding ids
    :type labels_to_ids: dict[str, str]
    :param learning_rate: the learning rate
    :type learning_rate: float
    :param num_epochs: the number of epochs
    :type num_epochs: int
    :param batch_size: the batch_size
    :type batch_size: int
    """

    experiment_name = f"{cf.model_config['name']}_e_{num_epochs}_bs_{batch_size}_lr_{learning_rate}"
    seq_weights_dir = os.path.join(cf.seq_weights_dir, experiment_name)
    model = cf.model_config['seq_model'].from_pretrained(seq_weights_dir,
                                                         num_labels=len(labels_to_ids),
                                                         return_dict=False)

    model.to(cf.device)

    seq_training.run_testing(model=model,
                             data_loader=testing_loader,
                             labels_to_ids=labels_to_ids,
                             model_dir=seq_weights_dir)


def train_val_seq():
    """ Do validation for Sequence Classification, saving the results """

    tokenizer = get_tokenizer()

    seq_weights = get_mlm_model(tokenizer=tokenizer)

    epochs_nums = cf.seq_hyperparameters["epochs"]
    batch_sizes = cf.seq_hyperparameters["batch_sizes"]
    learning_rates = cf.seq_hyperparameters["learning_rates"]

    for batch_size in batch_sizes:
        training_loader, validation_loader, _, labels_to_ids = seq_processing.process_data(dataset_dir=cf.seq_data_dir,
                                                                                           tokenizer=tokenizer,
                                                                                           batch_size=batch_size)
        for epochs_num in epochs_nums:
            for learning_rate in learning_rates:
                train_seq(training_loader=training_loader,
                          labels_to_ids=labels_to_ids,
                          weights=seq_weights,
                          num_epochs=epochs_num,
                          learning_rate=learning_rate,
                          batch_size=batch_size)
                test_seq(testing_loader=validation_loader,
                         labels_to_ids=labels_to_ids,
                         learning_rate=learning_rate,
                         num_epochs=epochs_num,
                         batch_size=batch_size)


def train_test_seq(epochs=cf.seq_epochs, learning_rate=cf.seq_learning_rate,
                   batch_size=cf.seq_batch_size):
    """ Do training and testing for Sequece Classification

    :param epochs: the number of epochs, defaults to cf.seq_epochs
    :type epochs: int, optionsl
    :param learning_rate: the learning rate, defaults to cf.seq_learning_rate
    :type learning_rate: float, optional
    :param batch_size: the batch size, defaults to cf.seq_batch_size
    :type batch_size: int, optional
    """

    tokenizer = get_tokenizer()

    seq_weights = get_mlm_model(tokenizer=tokenizer)

    training_loader, validation_loader, testing_loader, labels_to_ids = seq_processing.process_data(
        dataset_dir=cf.seq_data_dir,
        tokenizer=tokenizer,
        batch_size=batch_size)

    train_val_loader = DataLoader(
        ConcatDataset([training_loader.dataset, validation_loader.dataset]),
        shuffle=True,
        batch_size=training_loader.batch_size)

    train_seq(training_loader=train_val_loader,
              labels_to_ids=labels_to_ids,
              weights=seq_weights,
              num_epochs=epochs,
              learning_rate=learning_rate,
              batch_size=batch_size)
    test_seq(testing_loader=testing_loader,
             labels_to_ids=labels_to_ids,
             learning_rate=learning_rate,
             num_epochs=epochs,
             batch_size=batch_size)


def inference_ner(report):
    """ Do named entity recognition inference on a report

    :param report: the input report
    :type report: str
    """

    tokenizer = get_tokenizer()

    _, _, _, labels_to_ids = ner_processing.process_data(dataset_dir=cf.ner_data_dir,
                                                         tokenizer=tokenizer,
                                                         batch_size=cf.ner_batch_size)

    experiment_name = f"{cf.model_config['name']}_e_{cf.ner_epochs}_bs_{cf.ner_batch_size}_lr_{cf.ner_learning_rate}"
    ner_weights_dir = os.path.join(cf.ner_weights_dir, experiment_name)

    model = cf.model_config['ner_model'].from_pretrained(ner_weights_dir,
                                                         num_labels=len(labels_to_ids),
                                                         return_dict=False)
    model.to(cf.device)

    for sentence in report.strip().split('.'):
        if sentence != '':
            words, word_predictions = ner_training.run_inference(sentence=sentence,
                                                                 tokenizer=tokenizer,
                                                                 model=model,
                                                                 labels_to_ids=labels_to_ids)

            print(words)
            print(word_predictions)


def inference_seq(report):
    """ Do text classification inference on a report

    :param report: the input report
    :type report: str
    """

    tokenizer = get_tokenizer()

    _, _, _, labels_to_ids = seq_processing.process_data(dataset_dir=cf.seq_data_dir,
                                                         tokenizer=tokenizer,
                                                         batch_size=cf.seq_batch_size)

    experiment_name = f"{cf.model_config['name']}_e_{cf.seq_epochs}_bs_{cf.seq_batch_size}_lr_{cf.seq_learning_rate}"
    seq_weights_dir = os.path.join(cf.seq_weights_dir, experiment_name)

    model = cf.model_config['seq_model'].from_pretrained(seq_weights_dir,
                                                         num_labels=len(labels_to_ids),
                                                         return_dict=False)
    model.to(cf.device)

    for sentence in report.strip().split('.'):
        if sentence != '':
            sentence_prediction = seq_training.run_inference(sentence=sentence,
                                                             tokenizer=tokenizer,
                                                             model=model,
                                                             labels_to_ids=labels_to_ids)

            print(sentence)
            print(sentence_prediction)


if __name__ == '__main__':
    torch.cuda.empty_cache()

    if not os.path.exists(cf.weights_dir):
        os.makedirs(cf.weights_dir)

    if not os.path.exists(cf.tok_weights_dir):
        os.makedirs(cf.tok_weights_dir)

    if not os.path.exists(cf.mlm_weights_dir):
        os.makedirs(cf.mlm_weights_dir)

    # train_val_ner()
    # train_val_seq()

    # after validation, set the best hyperparameters values in the config.py
    train_test_ner()
    train_test_seq()

    # test_report_path = os.path.join(cf.tok_data_dir, "report_2088_1659029.txt")
    # with open(test_report_path, 'r', encoding='utf-8') as report_file:
    #     test_report = report_file.readlines()[0]

    # inference_ner(report=test_report)
    # inference_seq(report=test_report)
