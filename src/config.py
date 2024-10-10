import torch
import os
from transformers import BertTokenizer, BertForMaskedLM, BertForTokenClassification, BertForSequenceClassification, \
    BertConfig, RobertaForSequenceClassification, RobertaForTokenClassification, RobertaForMaskedLM, RobertaConfig, \
    RobertaTokenizer

# -------------------------------- PARAMS -------------------------------

device = 'cpu'  #torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# --------------------------------- MODELS ------------------------------

bert_config = {
    'name': 'bert',
    'tok_model': BertTokenizer,
    'tok_prefix': '##',
    'conf_model': BertConfig,
    'mlm_model': BertForMaskedLM,
    'ner_model': BertForTokenClassification,
    'seq_model': BertForSequenceClassification,
    'tok_train': False,
    'mlm_train': False,
}

roberta_config = {
    'name': 'roberta',
    'tok_model': RobertaTokenizer,
    'tok_prefix': 'Ġ',
    'conf_model': RobertaConfig,
    'mlm_model': RobertaForMaskedLM,
    'ner_model': RobertaForTokenClassification,
    'seq_model': RobertaForSequenceClassification,
}

model_config = bert_config

# ---------------------------- TOKENIZER PARAMS -------------------------

max_length = 64
min_frequency = 1
vocab_size = 30_522

# ------------------------------- MLM PARAMS ----------------------------

mlm_epochs = 3
mlm_batch_size = 4
mlm_learning_rate = 1e-4

# ------------------------------- NER PARAMS ----------------------------

ner_train_size = 0.7
ner_valid_size = 0.1
ner_test_size = 0.2

ner_epochs = 13
ner_batch_size = 4
ner_learning_rate = 1e-05

ner_max_grad_norm = 10

ner_hyperparameters = {
    "epochs": [5, 8, 13],
    "batch_sizes": [4, 8, 16],
    "learning_rates": [1e-3, 1e-4, 1e-5]
}

# ------------------------------- SEQ PARAMS ----------------------------

seq_train_size = 0.7
seq_valid_size = 0.1
seq_test_size = 0.2

seq_epochs = 13
seq_batch_size = 8
seq_learning_rate = 1e-5

seq_max_grad_norm = 10

seq_hyperparameters = {
    "epochs": [5, 8, 13],
    "batch_sizes": [4, 8, 16],
    "learning_rates": [1e-3, 1e-4, 1e-5]
}

# balanced classes threshold
imbalance_threshold = 25.0

# -------------------------------- PATHS --------------------------------

bert_pretrained_weights_ita = 'dbmdz/bert-base-italian-uncased'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_ID = ''

# weights paths
model_dir = os.path.join(ROOT_DIR, 'model')
experiment_dir = os.path.join(model_dir, EXPERIMENT_ID)

weights_dir = os.path.join(experiment_dir, 'weights')
tok_weights_dir = os.path.join(weights_dir, 'tok')
mlm_weights_dir = os.path.join(weights_dir, 'mlm')
ner_weights_dir = os.path.join(weights_dir, 'ner')
seq_weights_dir = os.path.join(weights_dir, 'seq')

# confusion matrices paths
conf_mats_dir = os.path.join(experiment_dir, 'conf_mats')
ner_conf_mats_dir = os.path.join(conf_mats_dir, 'ner')
seq_conf_mats_dir = os.path.join(conf_mats_dir, 'seq')

# reports paths
reports_dir = os.path.join(experiment_dir, 'reports')
mlm_reports_dir = os.path.join(reports_dir, 'mlm')
ner_reports_dir = os.path.join(reports_dir, 'ner')
seq_reports_dir = os.path.join(reports_dir, 'seq')

# data paths
data_dir = os.path.join(ROOT_DIR, 'data')
tok_data_dir = os.path.join(data_dir, 'tok')
mlm_data_dir = os.path.join(data_dir, 'tok')
ner_data_dir = os.path.join(data_dir, 'ner')
seq_data_dir = os.path.join(data_dir, 'seq')

# dataset paths
ner_train_path = os.path.join(ner_data_dir, 'ner_train_data.csv')
ner_valid_path = os.path.join(ner_data_dir, 'ner_valid_data.csv')
ner_test_path = os.path.join(ner_data_dir, 'ner_test_data.csv')

seq_train_path = os.path.join(seq_data_dir, 'seq_train_data.csv')
seq_valid_path = os.path.join(seq_data_dir, 'seq_valid_data.csv')
seq_test_path = os.path.join(seq_data_dir, 'seq_test_data.csv')

# classes paths
ner_l2id_path = os.path.join(ner_data_dir, 'labels_to_ids.json')
seq_l2id_path = os.path.join(seq_data_dir, 'labels_to_ids.json')
