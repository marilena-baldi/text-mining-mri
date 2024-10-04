import os
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt


def plot_conf_mat(cf_matrix, classes, conf_mat_path):
    """ Plot and save the confusion matrix

    :param cf_matrix: the confusion matrix
    :type cf_matrix: ndarray
    :param classes: the list of classes
    :type classes: List[str]
    :param conf_mat_path: the path where to save the confusion matrix
    :type conf_mat_path: str
    """

    conf_mat_dir = os.path.dirname(conf_mat_path)
    if not os.path.exists(conf_mat_dir):
        os.makedirs(conf_mat_dir)

    df_cm = pd.DataFrame(cf_matrix, index=classes, columns=classes)
    sns.set(rc={'figure.figsize': (10, 10)})
    sns.heatmap(df_cm, fmt='.2f', annot=True, cmap="PuBu")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.savefig(conf_mat_path)
    plt.clf()


def print_train_report(report_path, loss, accuracy, epoch_num):
    """ Print and save a report for training

    :param report_path: the path of the report to save
    :type report_path: str
    :param loss: the loss value
    :type loss: float
    :param accuracy: the accuracy value
    :type accuracy: float
    :param epoch_num: the current epoch (for training)
    :type epoch_num: int
    """

    reports_dir = os.path.dirname(report_path)
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    with open(report_path, 'a+', encoding="utf-8") as report_file:
        report_file.write(f'Training - epoch {epoch_num}\t')
        report_file.write(f'Loss: {loss}\t')
        report_file.write(f'Accuracy: {accuracy}\n')


def print_test_report(report_path, loss, accuracy, metrics_report):
    """ Print and save a report for training

    :param report_path: the path of the report to save
    :type report_path: str
    :param loss: the loss value
    :type loss: float
    :param accuracy: the accuracy value
    :type accuracy: float
    :param metrics_report: the report with the metrics
    :type metrics_report: str
    """

    reports_dir = os.path.dirname(report_path)
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    with open(report_path, 'a+', encoding="utf-8") as report_file:
        report_file.write('Testing\t')
        report_file.write(f'Loss: {loss}\t')
        report_file.write(f'Accuracy: {accuracy}\t')
        report_file.write(f'Report: \n {metrics_report}\n')
