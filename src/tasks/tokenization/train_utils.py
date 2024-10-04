import glob
import os
from tokenizers.implementations import BertWordPieceTokenizer, ByteLevelBPETokenizer

from src import config as cf


def train_wp(data_dir, model_save_path):
    """ Train a WordPieceTokenizer on txt files and save the vocabulary.

    :param data_dir: path to a directory containing the txt files from which to build the vocabulary
    :type data_dir: str
    :param model_save_path: path to a directory in which to save the vocabulary
    :type model_save_path: str
    """

    files = glob.glob(os.path.join(data_dir, '*.txt'))

    tokenizer = BertWordPieceTokenizer(lowercase=True, handle_chinese_chars=False)
    tokenizer.train(files=files,
                    vocab_size=cf.vocab_size,
                    min_frequency=cf.min_frequency,
                    show_progress=True)

    tokenizer.save_model(model_save_path)


def train_bl(data_dir, model_save_path):
    """ Train a ByteLevelBPETokenizer on txt files and save the vocabulary.

    :param data_dir: path to a directory containing the txt files from which to build the vocabulary
    :type data_dir: str
    :param model_save_path: path to a directory in which to save the vocabulary
    :type model_save_path: str
    """

    files = glob.glob(os.path.join(data_dir, '*.txt'))

    tokenizer = ByteLevelBPETokenizer(lowercase=True)
    tokenizer.train(files=files,
                    vocab_size=cf.vocab_size,
                    min_frequency=cf.min_frequency,
                    show_progress=True,
                    special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

    tokenizer.save_model(model_save_path)
