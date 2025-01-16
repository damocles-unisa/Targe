import os

import pandas as pd
import torch

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm


def compute_score(df: pd.DataFrame, model: str):
    """
    Compute the cosine similarity score between the description and the recipe.

    Args:
        df (pd.Dataframe): input data
        model (str): name of the SentenceTransformer model
    Returns:
        pd.Dataframe: data with the cosine similarity score
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')

    model = SentenceTransformer(model)
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            df.at[index, 'score'] = cosine_similarity([model.encode(row['desc'])],
                                                              [model.encode(row['recipe'])])
        except Exception as e:
            print(f'Error at index {index}: {e}')
            print(f'desc: {row["desc"]}')
            print(f'recipe: {row["recipe"]}')
            df.at[index, 'score'] = 0

    return df

def save_dataframe(df, output, filename, columns, instruction):
    """
    Save a dataframe to CSV and JSON with specified columns and added instruction.

    Args:
    df (pd.DataFrame): Dataframe to process and save.
    output (str): Output directory path.
    filename (str): Base filename for saving files (without extension).
    columns (list): List of columns to include in the saved dataframe.
    instruction (str): Instruction text to add to the dataframe.
    """
    save_path_csv = os.path.join(output, f'{filename}.csv')
    save_path_json = os.path.join(output, f'{filename}.json')
    dataset_path = 'data/dataset'
    # Select specified columns and add the instruction
    selected_df = df[columns].copy()
    selected_df['instruction'] = instruction

    selected_df.rename(columns={'desc': 'input', 'recipe': 'output', 'action': 'output', 'trigger': 'output'}, inplace=True)

    selected_df.to_json(save_path_json, orient='records', indent=4)
    selected_df.to_csv(save_path_csv, index=False)

    gold_df = selected_df[selected_df['score'] >= 0.7].reset_index(drop=True)
    noisy_df = selected_df[selected_df['score'] < 0.7].reset_index(drop=True)

    gold_df_test = gold_df.sample(n=1900, random_state=42)
    noisy_df_test = noisy_df.sample(n=1900, random_state=42)

    train_set = selected_df[~selected_df.index.isin(gold_df_test.index) & ~selected_df.index.isin(noisy_df_test.index)]

    train_set.to_csv(os.path.join(dataset_path, f'train_{filename}.csv'), index=False)
    train_set.to_json(os.path.join(dataset_path, f'train_{filename}.json'), orient='records', indent=4)
    gold_df_test.to_csv(os.path.join(dataset_path, 'gold', f'test_{filename}.csv'), index=False)
    gold_df_test.to_json(os.path.join(dataset_path, 'gold', f'test_{filename}.json'), orient='records', indent=4)
    noisy_df_test.to_csv(os.path.join(dataset_path, 'noisy', f'test_{filename}.csv'), index=False)
    noisy_df_test.to_json(os.path.join(dataset_path, 'noisy', f'test_{filename}.json'), orient='records', indent=4)


def save_full_dataset(cleaned_df, output):
    full_dataset = cleaned_df[cleaned_df['score'] > 0.2].reset_index(drop=True)
    full_dataset['instruction'] = ("From the description of a rule: identify the 'trigger', identify the 'action', "
                                   "write a IF 'trigger' THEN 'action' rule.")
    full_dataset.drop(columns=['trigger', 'action'], inplace=True)
    full_dataset.rename(columns={'desc': 'input', 'recipe': 'output'},
                        inplace=True)
    full_dataset.to_csv(os.path.join(output, 'full_recipe.csv'), index=False)
    full_dataset.to_json(os.path.join(output, 'full_recipe.json'), orient='records', indent=4)

    train_df, test_df = train_test_split(full_dataset, test_size=0.2)
    dataset_path = 'data/dataset'
    train_df.to_csv(os.path.join(dataset_path, 'train_full_recipe.csv'), index=False)
    train_df.to_json(os.path.join(dataset_path, 'train_full_recipe.json'), orient='records', indent=4)
    test_df.to_csv(os.path.join(dataset_path, 'test_full_recipe.csv'), index=False)
    test_df.to_json(os.path.join(dataset_path, 'test_full_recipe.json'), orient='records', indent=4)
