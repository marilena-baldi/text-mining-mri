import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from collections import defaultdict
import streamlit as st

from tasks.named_entity_recognition.processing_utils import process_data
from tasks.named_entity_recognition.train_test_utils import run_inference
import config as cf


@st.cache
def load_model():
    """ Load the tokenizer, the Named Entity Recognition model and the mapping dict with ids and
    labels. """

    tokenizer = cf.model_config['tok_model'].from_pretrained(cf.bert_pretrained_weights_ita,
                                                             max_len=cf.max_length)

    _, _, _, labels_to_ids = process_data(dataset_dir=cf.ner_data_dir,
                                          tokenizer=tokenizer,
                                          batch_size=cf.ner_batch_size)

    experiment_name = f"{cf.model_config['name']}_e_{cf.ner_epochs}_bs_{cf.ner_batch_size}_lr_{cf.ner_learning_rate}"
    ner_weights_dir = os.path.join(cf.ner_weights_dir, experiment_name)
    model = cf.model_config['ner_model'].from_pretrained(ner_weights_dir,
                                                         num_labels=len(labels_to_ids),
                                                         return_dict=False)
    model.to(cf.device)

    return tokenizer, model, labels_to_ids


def get_predictions(in_text, ner_model, tokenizer, labels_to_ids):
    """ Use a Tokenizer and a Named Entity Recognition model (along with the labels to ids mapping dict)
    to get entity tags of the input text.

    :param in_text: the input text for which to predict entities
    :type in_text: str
    :param ner_model: the Named Entity Recognition model
    :type ner_model: transformers.BertForTokenClassification.model |
    transformers.RobertaForTokenClassification.model
    :param tokenizer: the tokenizer
    :type tokenizer: transformers.BertTokenizer | transformers.RobertaTokenizer
    :param labels_to_ids: the labels and the corresponding ids
    :type labels_to_ids: dict[str, str]
    :return: the sentences and the tags
    :rtype: (List[str], List[List[str]])
    """

    sentences = []
    tags = []
    for sentence in in_text.strip().replace(';', '.').split('.'):
        words, word_predictions = run_inference(sentence=sentence,
                                                tokenizer=tokenizer,
                                                model=ner_model,
                                                labels_to_ids=labels_to_ids)

        if word_predictions:
            sentences.append(words)
            tags.append(word_predictions)

    return sentences, tags


def show_predicted_output(sentences_lists, tags_lists):
    """ Show a text with consecutively words and tags obtained with Named Entity Recognition for each
    input sentence.

    :param sentences_lists: a list of sentences
    :type sentences_lists: List[str]
    :param tags_lists: the list of the corresponding tags
    :type tags_lists: List[str]
    """

    for sentence, tags in zip(sentences_lists, tags_lists):
        words = sentence.split()

        assert len(words) == len(tags)

        output = ''
        for word, tag in zip(words, tags):
            output += str(word) + ' [' + str(tag) + '] '

        st.write(output)


def show_structured_output(sentences_lists, tags_lists):
    """ Display a json object with keywords and lists of tags obtained with Named Entity Recognition

    :param sentences_lists: a list of sentences
    :type sentences_lists: List[str]
    :param tags_lists: the list of the corresponding tags
    :type tags_lists: List[str]
    """

    tags_mapping = {'B-CLI': 'finding', 'I-CLI': 'finding', 'B-CER': 'certainty',
                    'I-CER': 'certainty', 'B-OBS': 'observation', 'I-OBS': 'observation',
                    'B-LOC': 'location', 'I-LOC': 'location', 'B-CHS': 'modifier',
                    'I-CHS': 'modifier', 'B-CHG': 'modifier', 'I-CHG': 'modifier',
                    'B-SIZ': 'modifier', 'I-SIZ': 'modifier', 'B-GRD': 'modifier',
                    'I-GRD': 'modifier'}

    full_template = {}
    for sentence, tags in zip(sentences_lists, tags_lists):
        template = defaultdict(list)

        words = sentence.split()

        assert len(words) == len(tags)

        for word, tag in zip(words, tags):

            if tag != 'O':
                try:
                    if tag.startswith('B'):
                        template[tags_mapping[tag]].append(word)
                    elif tag.startswith('I'):
                        template[tags_mapping[tag]][-1] += ' ' + word

                except IndexError:
                    template[tags_mapping[tag]].append(word)

        full_template[sentence] = template

    st.json(full_template)


if __name__ == '__main__':
    tokenizer, model, labels_to_ids = load_model()

    st.title('Reports Structuring')

    uploaded_file = st.file_uploader('Choose a file', type='txt')

    report_text = ''
    if uploaded_file:
        bytes_data = uploaded_file.getvalue().decode('utf-8')
        st.write(bytes_data)
        report_text = bytes_data

    sentences_lists, tags_lists = get_predictions(in_text=report_text,
                                                  tokenizer=tokenizer,
                                                  ner_model=model,
                                                  labels_to_ids=labels_to_ids)

    if st.button('Show text prediction'):
        show_predicted_output(sentences_lists=sentences_lists, tags_lists=tags_lists)

    if st.button('Show template prediction'):
        show_structured_output(sentences_lists=sentences_lists, tags_lists=tags_lists)
