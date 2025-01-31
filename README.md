# IoT Rule Generation with Cross-View Contrastive Learning and Perplexity-Based Ranking

This repository contains the supplementary material for the paper "IoT Rule Generation with Cross-View Contrastive Learning and Perplexity-Based Ranking" submitted to IEEE Internet of Things Journal.

This material comprises the source codes useful for the repeatability of the experiments.

<div align="center">
    <img src="./docs/rule_example.svg" alt="TARGE LOGO" width="250" height="250" style="margin: 0 auto">
</div>

# Creators
Gaetano Cimino (gcimino@unisa.it), Vincenzo Deufemia (deufemia@unisa.it), and Mattia Limone (mattia.limone@mail.com)

# Overview
We propose TARGE (Trigger-Action Rule GEneration), a novel framework for generating IoT automation rules 
directly from natural language user intents. TARGE leverages Large Language Models (LLMs) to interpret user intents and employs cross-view contrastive learning to generate rule embeddings that capture TAP functionality and device relationships. Its ranking mechanism combines semantic consistency with LLM-derived perplexity to prioritize contextually coherent rules.

# TARGE Architecture
<div align="center">
    <img src="./docs/schema_gen_1.svg" alt="TARGE Architecture"   style="margin: 0 auto; background-color: white">
</div>

The process is divided into two primary modules: (a) Synthetic Rule Representation and (b) Rule Component Selection. The first module focuses on deriving the semantic representation of a synthetic rule. Synthetic rules are outputs produced by an LLM that are not directly usable as final outcomes because they may correspond to invalid functionalities within a Trigger-Action Platform (TAP). Specifically, the LLM processes the user's intent and translates it into a synthetic rule, where the components represent synthetic trigger/action pairs. These components are semantically encoded by two embedding models, which are specialized for processing triggers and actions, respectively. Using a cross-view contrastive learning strategy, the models embed trigger-action functionalities and channels into vectors, allowing for comprehensive semantic representation. Insetad, the second module, Rule Component Selection, identifies concrete components in a TAP by classifying trigger-action embeddings into specific channel categories. This module integrates a crawler to retrieve the trigger and action functionalities corresponding to the channels within the predicted categories. A ranking mechanism is employed to determine the most relevant trigger-action pair that aligns with the user's intent, using a similarity metric to compare the semantic embeddings of synthetic and actual TAP components. For scenarios requiring multiple recommendations, the module applies a perplexity-based ranking mechanism, combining LLM-derived perplexity scores with similarity measures to evaluate and prioritize trigger-action combinations. 

# Setting Environment
A requirements.txt file is included to install the necessary packages. To install them, simply run `pip install -r requirements.txt`.

# Pre-processing
To generate the dataset, run `python src/preprocessing/build_dataset.py`.
The script will create the dataset from the IFTTT platform and save it in the `data/dataset` directory.

To create the dataset for training the embedder, run `python src/preprocessing/build_embedder_dataset.py`.

# Training
To train the classifier, run `python src/train/train_classifier.py`.

To train the LLM, run `python src/train/train_llm.py`. The script includes arguments to define the training scope (rule, action, trigger) and the set (clear, unclear).

To train the embedder, run `python src/train/train_embedder.py`.

# Evaluation
The evaluation scripts are located in the `src/evaluate` directory. 
First, generate all the rules from the trained LLM by running `python src/evaluate/generate.py`. 
Then, use the following notebooks to evaluate the generated rules.

- `src/evaluate/evaluate_final.ipynb` is the notebook to evaluate TARGE on gold and noisy test sets
- `src/evaluate/evaluate_final_one_shot.ipynb` is the notebook to evaluate TARGE on the one-shot test set
- `src/evaluate/evaluate_final_fasttext.ipynb` is the notebook to evaluate TARGE with fastText embeddings
- `src/evaluate/evaluate_final_word2vec.ipynb` is the notebook to evaluate TARGE with word2vec embeddings

