import os
import pandas as pd
import re
import string
import numpy as np

from src import config as cf

reports = ["data_user1_2017.csv", "data_user2_2017.csv", "data_user3_2017.csv"]
report_csv_paths = [os.path.join(cf.data_dir, report) for report in reports]
reports_txt_dir = cf.tok_data_dir
ner_csv_out_dir = cf.data_dir
seq_csv_out_dir = cf.seq_data_dir

COLUMN_USER_ID = "IdUtente"
COLUMN_TEXT_ID = "IdTesto"
COLUMN_TEXT = "TestoRefertoStringa"


def process_sentence(sentence):
    """ Process a textual sentence performing cleaning operations and returning the cleaned sentence

    :param sentence: the input textual sentence
    :type sentence: str
    :return: the cleaned sentence
    :rtype: str
    """

    sentence = sentence.replace('-', ' ').replace(';.', '.').replace('.', '. ')
    sentence = re.sub(r'(tsrm).*', '', sentence)
    sentence = re.sub(r'((\.)+)', '.', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'\d{1,2}[/,:/\s]\d{1,2}[/,:/\s]\d{2,4}', 'xx/xx/xxxx', sentence)
    sentence = re.sub(r'\d+,\d+\s*us', 'xx/xx/xxxx', sentence)
    sentence = re.sub(r'(\d{1,2})(cm|mm|m)', r'\1 \2', sentence)
    sentence = re.sub(r'(\D+)\s+mm\s+(\D+)', r'\1 muscoli \2', sentence)
    sentence = re.sub(r'(\D+)\s+m\s+(\D+)', r'\1 muscolo \2', sentence)

    return sentence


def process_report(report):
    """ Process a textual report performing cleaning operations and returning the report as a list of cleaned sentences

    :param report: the textual report
    :type report: str
    :return: the cleaned report sentences
    :rtype: List[str]
    """

    report = report.lower()
    report = report.replace("art.ne", "articolazione").replace("r.m.", "rm").replace("dott.", "dott").\
        replace("mm.", "mm").replace("m.", "m").replace("u.s.", "us")

    report = re.sub(r'(\s)-(\s)', '', report)
    report = re.sub(r'\s*,\s*', ', ', report)
    report = re.sub(r'(\s):(\s)', ': ', report)

    report = re.sub(r'(\d)\s*\.\s*(\d)\s*([a-zA-Z]{2})', r'\1,\2 \3', report)
    report = re.sub(r'\.?\s*reperto rm\s*\.?', '. reperto rm .', report) if "reperto rm limitato" not in report else \
        re.sub(r'\.?\s*reperto rm\s*?', '. reperto rm ', report)
    report = re.sub(r'\.?\s*l\'indagine\s*', '. l\'indagine ', report)
    report = re.sub(r'\.?\s*tecnica d\'esame\s*?', '. tecnica d\'esame ', report)

    split_report_sentences = [sentence.strip() for sentence in re.split(r'[.;:\n]', report.strip()) if len(sentence) > 0]

    clean_report_sentences = list(map(process_sentence, split_report_sentences))

    clean_report_sentences = [sent.strip() + '. ' if not sent.endswith('. ') else sent
                              for sent in clean_report_sentences if len(sent) > 0]

    return clean_report_sentences


def extract_reports(csv_db_path, txt_dir):
    """ Extract, process and save reports from a csv as txt files to prepare data for building the tokenizer vocabulary

    :param csv_db_path: the path to the csv file with the reports
    :type csv_db_path: str
    :param txt_dir: the path to the dir in which to save reports as txt files
    :type txt_dir: str
    """

    df = pd.read_csv(csv_db_path, skipinitialspace=True, header=0, encoding='unicode_escape')

    for index, row in df.iterrows():
        user_id = row[COLUMN_USER_ID]
        text_id = row[COLUMN_TEXT_ID]
        text = row[COLUMN_TEXT]

        clean_sentences = process_report(report=text)

        with open(txt_dir + f'/report_{user_id}_{text_id}.txt', 'w') as f:
            f.writelines(clean_sentences)


def get_annotation_file(csv_db_path, csv_out_dir):
    """ Extract, process and save reports from a csv as a csv annotation file with sentences, words and O tags.

    :param csv_db_path: the path to the csv file with the reports
    :type csv_db_path: str
    :param csv_out_dir: the path to the output dir for the csv file for the annotation
    :type csv_out_dir: str
    """

    df = pd.read_csv(csv_db_path, skipinitialspace=True, header=0, encoding='unicode_escape')

    i, user_id = 0, 0
    grade = "I"
    sentence_num_list, word_list, tag_list, cls_list, user_ids, report_ids = [], [], [], [], [], []
    for _, row in df.iterrows():
        user_id = row[COLUMN_USER_ID]
        report_id = row[COLUMN_TEXT_ID]
        report = row[COLUMN_TEXT]

        sentences = process_report(report=report)
        for sentence in sentences:
            sentence = re.sub(r'(?<! )(?=[{}])|(?<=[{}])(?! )'.format(
                string.punctuation, string.punctuation), r' ', sentence)

            sentence_words = sentence.split()
            sentence_tags = ['O'] * len(sentence_words)

            sentence_num_list.append('Sentence: {}'.format(i))
            sentence_num_list.extend([''] * (len(sentence_words) - 1))

            cls_list.append(grade)
            cls_list.extend([''] * (len(sentence_words) - 1))

            user_ids.append(user_id)
            user_ids.extend([''] * (len(sentence_words) - 1))

            report_ids.append(report_id)
            report_ids.extend([''] * (len(sentence_words) - 1))

            word_list.extend(sentence_words)
            tag_list.extend(sentence_tags)

            i += 1

    df = pd.DataFrame(list(zip(user_ids, report_ids, sentence_num_list, word_list, tag_list, cls_list)),
                      columns=['UserId', 'ReportId', 'Sentence #', 'Word', 'Tag', 'Cls'])
    csv_out_path = os.path.join(csv_out_dir, f'dataset_{user_id}.csv')
    df.to_csv(csv_out_path, index=False)

def split_annotation_file(csv_db_path,tag_path):
    """
    Split annpotations by column (Tag for ner

    """

    df = pd.read_csv(csv_db_path, skipinitialspace=True, header=0, encoding='utf-8')
    data_cols = list(df.columns)
    for t in tag_path:
        data_cols.remove(t)
    for t in tag_path:
        data_cols.append(t)
        df.filter(items=data_cols).to_csv(os.path.join(tag_path[t],os.path.split(csv_db_path)[1]),index=False)
        df['ReportId'] = df['ReportId'].astype('Int64')
        df['UserId'] = df['UserId'].astype('Int64')
        data_cols.remove(t)


def read_seq_csv(csv_path):
    df = pd.read_csv(csv_path, skipinitialspace=True, header=0, encoding='utf-8', keep_default_na=False)
    return df

def save_seq_csv(df,csv_path):
    df['ReportId'] = df['ReportId'].astype('Int64')
    df['UserId'] = df['UserId'].astype('Int64')
    df.to_csv(csv_path, index=False)


def perform_checks(csvs):
    """
    do some controls and transformation on csvs

    """
    bpath= 'C:\\progetti\\sdn\\src\\data\\seq\\'
    for csv in csvs:
        # control nan rows on UserId NOT NULL - CHECK LABELLING

        csv_path = bpath + csv
        df = read_seq_csv(csv_path)
        uniqueCls=df.Cls.unique()
        labels_to_ids = {}
        labels_to_ids.update({k: v for v, k in enumerate(uniqueCls)})
        nans = df.loc[df['UserId'].notnull() & df['Cls'].isna()].any(axis=1)
        print(csv +' : '+str(nans.sum()))
        print(labels_to_ids)
        if nans.sum()>0:
            print(df[df['UserId'].notnull() & df['Cls'].isna()].head(10))
        for index, row in df.iterrows():
            if row['UserId']=='':
                continue
            if row['Cls'] not in ['NA','I','II','III','IV']:
                print('wrong Cls line at '+str(index))

    csvs = ['user1.csv', 'user2.csv', 'user3.csv']

    for csv in csvs:
        csv_path = bpath + csv
        df = read_seq_csv(csv_path)
        mask = (df['UserId'].notnull() & df['Cls'].isna())
        df.loc[mask,'Cls']='NA'
        save_seq_csv(df,csv_path)
if __name__ == '__main__':
    if not os.path.exists(cf.tok_data_dir):
        os.makedirs(cf.tok_data_dir)

    if not os.path.exists(cf.ner_data_dir):
        os.makedirs(cf.ner_data_dir)

    tag_path={}
    tag_path['Tag'] = ner_csv_out_dir + "\\ner"
    tag_path['Cls'] = seq_csv_out_dir
    for report_csv_path in report_csv_paths:
        # extract_reports(csv_db_path=report_csv_path, txt_dir=reports_txt_dir)
        get_annotation_file(csv_db_path=report_csv_path, csv_out_dir=ner_csv_out_dir)
        #split_annotation_file(csv_db_path=report_csv_path,tag_path=tag_path)