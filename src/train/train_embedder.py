import os
from tqdm.autonotebook import tqdm
import pandas as pd
import ast
import numpy as np
from datasets import Dataset

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator
import warnings
import wandb

wandb.login(key="")

use_wandb = False
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def get_sentences_dataset(df):
    anchor = []
    positive = []
    negative = []

    for index, row in tqdm(df.iterrows(), total=len(df)):
        # Extract the list from the 'column' cell
        a = row['input']
        p_list = np.array(ast.literal_eval(row['positive']))
        n_list = np.array(ast.literal_eval(row['negative']))

        for i in range(len(p_list)):
            a_p = p_list[i]
            a_n = n_list[i]
            anchor.append(a)
            positive.append(a_p)
            negative.append(a_n)

            a_p = p_list[i]
            a_n = n_list[-1 - i]
            anchor.append(a)
            positive.append(a_p)
            negative.append(a_n)

    return {'anchor': anchor, 'positive': positive, 'negative': negative}


if __name__ == '__main__':
    output = 'data/dataset/embedder'

    name = 'NAME'
    model = SentenceTransformer(
        f"{name}",
        model_kwargs={'attn_implementation': 'flash_attention_2'}
    )
    sets = ['trigger','action']
    targets = ['channel']

    for set in sets:
        for target in targets:
            print(f"Training {name} on {set} {target} ")
            wandb_run_name = f"{set}_{target}_embedder_training"

            if target == 'both':
                train_set = pd.read_csv(os.path.join(output, f'train_{set}_channel.csv'))
                test_set = pd.read_csv(os.path.join(output, f'test_{set}_channel.csv'))
                train_set = pd.concat([
                    train_set,
                    pd.read_csv(os.path.join(output, f'train_{set}_function.csv'))
                ], ignore_index=True)

                test_set = pd.concat([
                    test_set,
                    pd.read_csv(os.path.join(output, f'test_{set}_function.csv'))
                ], ignore_index=True)
            else:
                train_set = pd.read_csv(os.path.join(output, f'train_{set}_{target}.csv'))
                test_set = pd.read_csv(os.path.join(output, f'test_{set}_{target}.csv'))

            train_dataset = Dataset.from_dict(get_sentences_dataset(train_set))
            test_dataset = Dataset.from_dict(get_sentences_dataset(test_set))
            print(f"Train Dataset: ", train_dataset)
            print(f"Test Dataset: ", test_dataset)

            loss = MultipleNegativesRankingLoss(model)

            args = SentenceTransformerTrainingArguments(
                output_dir=f"models/{name}/{set}_{target}_{name}_embedder",
                num_train_epochs=1,
                learning_rate=2e-5,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                warmup_ratio=0.1,
                fp16=True,
                bf16=False,
                batch_sampler=BatchSamplers.NO_DUPLICATES,
                save_strategy="steps",
                save_steps=12937,
                logging_steps=12937,
                report_to="wandb" if use_wandb else None,
                run_name=wandb_run_name if use_wandb else None,
            )

            trainer = SentenceTransformerTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                loss=loss
            )
            trainer.train()

            model.save_pretrained(f"models/{name}/{set}_{target}_{name}_embedder")

            test_evaluator = TripletEvaluator(
                anchors=test_dataset["anchor"],
                positives=test_dataset["positive"],
                negatives=test_dataset["negative"],
                name="test",
            )
            print(test_evaluator(model))
            print(f"Training {name} on {set} {target} finished")