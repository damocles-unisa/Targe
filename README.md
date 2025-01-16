<div align="center">
    <img src="./docs/rule_example.svg" alt="TARGE LOGO" width="250" height="250" style="margin: 0 auto">
    <h1 align="left">Generating Executable IoT Automation Rules from User Intents.</h1>
</div>


This is a replication package for our paper titled **TARGE: Generating Executable IoT Automation Rules from User Intents**. 

# Abstract

# Architecture

# Important Files
- `src/preprocessing/build_dataset.py` is the script to generate the dataset from the IFTTT and Recipe67k crawled data
- `src/preprocessing/build_embedder_dataset.py` is the first script to generate the dataset for the Embedder finetuning
- `src/train/train_classifier.py` is the training script for the classifier
- `src/train/train_llm.py` is the training script for the LargeLanguage Model
- `src/train/train_embedder.py` is the training script for the Embedder
- `src/evaluate/generate.py` is the script to generate the rules from the trained language models
- `src/evaluate/evaluate_llm.ipynb` is the notebook to evaluate the Large Language Models in ICG an CRG mode
- `src/evaluate/evaluate_final.ipynb` is the notebook to evaluate the generated rules
- `src/evaluate/evaluate_final_one_shot.ipynb` is the notebook to evaluate the generated rules on one-shot test set
- `src/evaluate/evaluate_final_fasttext.ipynb` is the notebook to evaluate the generated rules with FastText embeddings
- `src/evaluate/evaluate_final_word2vec.ipynb` is the notebook to evaluate the generated rules with word2vec embeddings

# Setting Environment
We provide  a requirements.txt file to install the necessary packages. You can install the required packages by running `pip install -r requirements.txt`.

# Preprocessing
To generate the dataset, run `python src/preprocessing/build_dataset.py`. The script will generate the dataset from the IFTTT and Recipe67k crawled data. The generated dataset will be saved in the `data/dataset` directory.

# Training
To train the classifier, run `python src/train/train_classifier.py`.
To train the Large Language Model, run `python src/train/train_llm.py`. args are provided in the script to specify the scope of the training (recipe, action, trigger) and the set (clear, unclear)
To train the Embedder, run `python src/train/train_embedder.py`.

# Inference
To generate the rules from the trained language models, run `python src/evaluate/generate.py`. The generated rules will be saved in the `data/generated` directory.
To evaluate the generated rules, run `python src/evaluate/evaluate.py`. The script will compute the metrics for the generated rules.


# Checkpoints and Result Artefacts
We release our model checkpoints and the corresponding inference results [here](to-add).

# Authors
- [Mattia Limone](https://www.linkedin.com/in/mattia-limone/)
- [Gaetano Cimino](https://www.linkedin.com/in/gaetano-cimino-6411a7174/)
- [Vincenzo Deufemia](https://docenti.unisa.it/005087/home)
