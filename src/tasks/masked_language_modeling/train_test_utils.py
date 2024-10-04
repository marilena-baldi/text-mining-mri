import os
import numpy as np
from transformers import AdamW
from tqdm import tqdm
import torch

from src.tasks.results_utils import print_test_report, print_train_report
from src import config as cf


def run_training(model, data_loader, num_epochs, learning_rate, model_save_dir):
    """ Train a model for Masked Language Modeling and save the weights.

    :param model: the model to train
    :type model: transformers.BertForMaskedLM.model | transformers.RobertaForMaskedLM.model
    :param data_loader: the data loader object with the training data
    :type data_loader: torch.utils.data.DataLoader
    :param num_epochs: the number or training epochs
    :type num_epochs: int
    :param learning_rate: the learning rate
    :type learning_rate: float
    :param model_save_dir: the path to the directory to save the weights
    :type model_save_dir: str
    """

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    optimizer = AdamW(params=model.parameters(), lr=learning_rate)
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

            accuracy += flat_accuracy(predictions=logits, labels=label_ids)

            output[0].backward()
            optimizer.step()

            loop.set_description(f'Epoch {i}')
            loop.set_postfix(loss=output[0])

        loss = loss / len(data_loader)
        accuracy = accuracy / len(data_loader)

        print('Training loss: ', loss)
        print('Training accuracy: ', accuracy)

        experiment_name = os.path.basename(os.path.normpath(model_save_dir))

        report_path = os.path.join(cf.mlm_reports_dir, experiment_name + "_train.txt")
        print_train_report(report_path=report_path,
                           loss=loss,
                           accuracy=accuracy,
                           epoch_num=i)

    model.save_pretrained(model_save_dir)


def run_testing(model, data_loader, model_dir):
    """ Test a model for Masked Language Modeling.

    :param model: the model to test
    :type model: transformers.BertForMaskedLM.model | transformers.RobertaForMaskedLM.model
    :param data_loader: DataLoader object with the test data
    :type data_loader: torch.utils.data.DataLoader
    :param model_dir: the path to the model folder
    :type model_dir: str
    """

    model.eval()

    loss, accuracy = 0, 0
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

            accuracy += flat_accuracy(predictions=logits, labels=label_ids)

            loop.set_description(f'Sample {i}')
            loop.set_postfix(loss=output[0])

    loss = loss / len(data_loader)
    accuracy = accuracy / len(data_loader)

    print('Test Loss: ', loss)
    print('Test Accuracy: ', accuracy)

    experiment_name = os.path.basename(os.path.normpath(model_dir))
    report_path = os.path.join(cf.mlm_reports_dir, experiment_name + "_test.txt")
    print_test_report(report_path=report_path,
                      loss=loss,
                      accuracy=accuracy,
                      metrics_report='')


def flat_accuracy(predictions, labels):
    """ Evaluate the accuracy for non flatten arrays.

    :param predictions: model's predictions
    :type predictions: torch.Tensor
    :param labels: model's labels
    :type labels: torch.Tensor
    :return: the accuracy
    :rtype: float
    """

    flatten_predictions = np.argmax(predictions, axis=-1).flatten()
    flatten_labels = labels.flatten()

    accuracy = np.sum(
        np.asarray(flatten_predictions) == np.asarray(flatten_labels)
        ) / len(flatten_labels)

    return accuracy
