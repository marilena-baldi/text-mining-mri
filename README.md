# Text mining mri

- [Description](#description)
- [Project Structure](#project-structure)
- [Getting started](#getting-started)
  - [Container](#container)
    - [Usage](#usage)
  - [Local](#local)
    - [Usage](#usage-1)
- [References](#references)


## Description
This is a machine learning project on report structuring [[1]](#1) and grade classification.
The goals are:
- to extract textual clinical entities from medical reports with the aim of structuring them with a template;
- to determine if a medical report contains evidences of osteoarthritis and to what grade.

For both tasks, the results were obtained by doing transfer learning of BERT [[2]](#2) based models, fine-tuning those made available by Hugging Face.

The structuring of the reports was achieved by classifying the tokenized text entities, using BERT for the Named Entity Recognition. 
Instead, the classification of the degree of osteoarthritis was performed by classifying the tokenized sentences of the text of the reports using BERT for the Sequence Classification.

## Project Structure
The `src` folder contains the core of the project, while the `stack` folder contains useful files such as the `Dockerfile` and the `docker-compose.yml` file.
The `Makefile` allows to quickly run the project using the containerized and orchestrated software.

Below is the organization of the source project:

- The `data` folder has:
  - the `ner` folder with the manually annotated data for Named Entity Recognition;
  - the `seq` folder with the manually annotated data for Sequence Classification;
  - the `tok` folder with the generated reports used to build the tokenizers' vocabularies and to possibly train BERT from scratch for Masked Language Modeling;
- The `data_utils` folder has:
  - the `data_processing.py` script with some functions to process and prepare data (mostly for the tokenizer, BERT for MLM and annotation files);
- The `model` folder has:
  - a folder to store the experiments confusion matrices;
  - a folder to store the experiments train and test results;
  - a folder to store the experiments weights;
- The `tasks` folder has:
  - the `masked_language_modeling` folder with a script to process data and a script to do training and testing;
  - the `named_entity_recognition` folder with a script to process data and a script to do training, testing and inference;
  - the `sequence_classification` folder with a script to process data and a script to do training, testing and inference;
  - the `tokenization` folder with a script to build tokenizers;
  - the `results_utils.py` script with functions to plot and save experiments results;
- The main folder has:
  - the `config.py` script to set up useful information such as paths and training parameters;
  - the `main.py` script to run the training, validation, testing and inference;
  - the `demo.py` script to run a streamlit-based demo web application;


## Getting started

Before starting, you can edit the `config.py` file and change the paths and folders names or training parameters.
If set to something, the `EXPERIMENT_ID` parameter allows to create a folder inside the `model` one and keep the runs results separate from each other.

You can also edit the `main.py` script to select what to do between training, validation, testing and inference.

### Container

To get started place in the base project folder and just type:
```sh
make build  # to build all docker containers

make shell  # to open a shell in the container
```

#### Usage

Container usage involves the same steps as [local usage](#usage-1) after entering the shell.

### Local

Make sure to have python 3.10 installed. 

Install the project requirements. Place in the `src` folder and type:
```sh
pip install -r requirements.txt
```

#### Usage

Place in the `src` folder.

If you want to try BERT-based models that involve 
building the tokenizer vocabulary (`tok_train: True` for `bert_config` in `config.py`) 
and training BERT for the MLM (`mlm_train: True` for `bert_config` in `config.py`), 
the first time it is launched, it is also necessary to generate the texts of the reports to be used as a dataset, 
therefore it is necessary to run:

```sh
python data_utils/data_processing.py
```

Otherwise, and in any case even after following the previous step, 
to run the `main.py` script and start the training/validation/testing/inference, type:

```sh
python main.py
```

To run the demo web application, type:

```sh
streamlit run demo.py
```

Before launching the demo make sure to have generated the neural network weights and have the `config.py` parameters set accordingly.

The demo web application will be available on `localhost:8501`.

## References
<a id="1">[1]</a> 
Kento Sugimoto et al. (2021). 
Extracting clinical terms from radiology reports with deep learning
https://www.sciencedirect.com/science/article/pii/S1532046421000587

<a id="2">[2]</a> 
Jacob Devlin et al. (2019) 
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
https://arxiv.org/pdf/1810.04805.pdf