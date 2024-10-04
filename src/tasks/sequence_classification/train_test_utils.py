import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from src.tasks.results_utils import plot_conf_mat, print_test_report, print_train_report
from src import config as cf


def run_training(model, data_loader, num_epochs, learning_rate, model_save_dir):
    """ Train a model for Sequence Classification and save the weights.

    :param model: the model to train
    :type model: transformers.BertForSequenceClassification.model |
    transformers.RobertaForSequenceClassification.model
    :param data_loader: DataLoader object with the training data
    :type data_loader: torch.utils.data.DataLoader
    :param num_epochs: number or training epochs
    :type num_epochs: int
    :param learning_rate: the learning rate
    :type learning_rate: float
    :param model_save_dir: path to the directory to save the weights
    :type model_save_dir: str
    """

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.train()

    for i in range(num_epochs):
        loss, accuracy = 0, 0

        loop = tqdm(data_loader, leave=True)
        for batch in loop:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(cf.device)
            attention_mask = batch['attention_mask'].to(cf.device)
            labels = batch['labels'].to(cf.device)

            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss += output[0].detach()

            logits = output[1].detach().cpu()
            label_ids = labels.detach().cpu()

            flatten_labels = label_ids.view(-1)
            predictions = torch.argmax(logits.view(-1, model.num_labels), dim=1)

            accuracy += accuracy_score(y_true=flatten_labels, y_pred=predictions)

            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=cf.seq_max_grad_norm
            )

            output[0].backward()
            optimizer.step()

            loop.set_description(f'Epoch {i}')
            loop.set_postfix(loss=output[0])

        loss = loss / len(data_loader)
        accuracy = accuracy / len(data_loader)

        print('Training loss: ', loss)
        print('Training accuracy: ', accuracy)

        experiment_name = os.path.basename(os.path.normpath(model_save_dir))
        report_path = os.path.join(cf.seq_reports_dir, experiment_name + "_train.txt")
        print_train_report(report_path=report_path,
                           loss=loss,
                           accuracy=accuracy,
                           epoch_num=i)

    model.save_pretrained(model_save_dir)


def run_testing(model, data_loader, labels_to_ids, model_dir):
    """ Test a model for Sequence Classification.

    :param model: the model to test
    :type model: transformers.BertForTokenClassification.model |
    transformers.RobertaForTokenClassification.model
    :param data_loader: DataLoader object with the test data
    :type data_loader: torch.utils.data.DataLoader
    :param labels_to_ids: the labels and the corresponding ids
    :type labels_to_ids: dict[str, str]
    :param model_dir: the path to the model folder
    :type model_dir: str
    """

    ids_to_labels = {v: k for k, v in labels_to_ids.items()}

    model.eval()

    loss, accuracy = 0, 0
    predictions_list, labels_list = [], []
    with torch.no_grad():
        loop = tqdm(data_loader, leave=True)

        for i, batch in enumerate(loop):
            input_ids = batch['input_ids'].to(cf.device)
            attention_mask = batch['attention_mask'].to(cf.device)
            labels = batch['labels'].to(cf.device)

            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss += output[0].detach()

            logits = output[1].detach().cpu()
            label_ids = labels.detach().cpu()

            flatten_labels = label_ids.view(-1)
            predictions = torch.argmax(logits.view(-1, model.num_labels), dim=1)

            labels_list.extend(flatten_labels)
            predictions_list.extend(predictions)

            accuracy += accuracy_score(y_true=flatten_labels, y_pred=predictions)

            loop.set_description(f'Sample {i}')
            loop.set_postfix(loss=output[0])

    labels = [ids_to_labels[el.item()] for el in labels_list]
    predictions = [ids_to_labels[el.item()] for el in predictions_list]

    loss = loss / len(data_loader)
    accuracy = accuracy / len(data_loader)

    print('Test Loss: ', loss)
    print('Test Accuracy: ', accuracy)

    report = classification_report(y_true=labels, y_pred=predictions, zero_division=True)
    print(report)

    conf_mat = confusion_matrix(labels, predictions, labels=list(labels_to_ids.keys()))
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    experiment_name = os.path.basename(os.path.normpath(model_dir))
    conf_mat_path = os.path.join(cf.seq_conf_mats_dir, experiment_name + '.svg')
    plot_conf_mat(cf_matrix=conf_mat,
                  classes=list(ids_to_labels.values()),
                  conf_mat_path=conf_mat_path)

    report_path = os.path.join(cf.seq_reports_dir, experiment_name + "_test.txt")
    print_test_report(report_path=report_path,
                      loss=loss,
                      accuracy=accuracy,
                      metrics_report=str(report))


def run_inference(sentence, tokenizer, model, labels_to_ids):
    """ Classify a sentence.

    :param sentence: sentence whose tokens have to be classified
    :type sentence: str
    :param tokenizer: the tokenizer
    :type tokenizer: transformers.BertTokenizer | transformers.RobertaTokenizer
    :param model: the model to test
    :type model: transformers.BertForSequenceClassification.model |
    transformers.RobertaForSequenceClassification.model
    :param labels_to_ids: the labels and the corresponding ids
    :type labels_to_ids: dict[str, dyt]
    :return: the words and the corresponding predictions
    :rtype: (str, List[str])
    """

    ids_to_labels = {v: k for k, v in labels_to_ids.items()}

    inputs = tokenizer(sentence,
                       padding='max_length',
                       truncation=True,
                       max_length=cf.max_length,
                       return_tensors="pt")

    ids = inputs["input_ids"].to(cf.device)
    mask = inputs["attention_mask"].to(cf.device)

    outputs = model(ids, mask)
    logits = outputs[0].detach().cpu()

    sentence_prediction = torch.argmax(logits.view(-1, model.num_labels), dim=1).item()
    sentence_prediction = ids_to_labels[sentence_prediction]

    return sentence_prediction
