import pandas as pd
import os
import numpy as np
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

from libs.preprocessing.utils import compute_score


def run(path: str, filename: str, model: str) -> pd.DataFrame:
    """
    Run the preprocessing pipeline.

    Args:
        path (str): path to the data file
        filename (str): name of the data file
        model (str): name of the SentenceTransformer model
        output (str): path to save the preprocessed data
    Returns:
        pd.DataFrame: preprocessed data
    """
    df = pd.read_csv(os.path.join(path, filename))
    print(df.isna().sum())
    # Add to the cleaned_df desc column the concat of the column title and desc of df with a space in between
    df['score'] = 0
    df['originalDesc'] = df['desc']
    df['desc'] = df['title'] + '. ' + df['desc']
    df['recipe'] = ('IF ' + df['triggerChannelTitle'] + ' ' + df['triggerTitle'] + ' THEN '
                    + df['actionChannelTitle'] + ' ' + df['actionTitle'])
    df['trigger'] = 'TRIGGER SERVICE: ' + df['triggerChannelTitle'] + ', TRIGGER EVENT: ' + df['triggerTitle']
    df['action'] = 'ACTION SERVICE: ' + df['actionChannelTitle'] + ', ACTION EVENT: ' + df['actionTitle']
    # add recipegen column that follows this format "Flickr <sep> Flickr.New_public_photos <sep> Twitter <sep> Twitter.Post_a_tweet"
    df['recipegen'] = (df["triggerChannelTitle"].str.replace(" ", "_")
                       + ' <sep> ' +
                       df["triggerChannelTitle"].str.replace(" ", "_") + '.' + df["triggerTitle"].str.replace(" ", "_")
                       + ' <sep> ' +
                       df["actionChannelTitle"].str.replace(" ", "_")
                       + ' <sep> ' +
                       df["actionChannelTitle"].str.replace(" ", "_") + '.' + df["actionTitle"].str.replace(" ", "_"))

    # Drop rows only where the relevant columns have NaN
    df.dropna(subset=['title', 'desc', 'triggerChannelTitle', 'triggerTitle', 'actionChannelTitle', 'actionTitle'],
              inplace=True)

    df.reset_index(drop=True, inplace=True)

    df = compute_score(df, model)

    df.to_csv(os.path.join(path, 'recipes_scored.csv'), index=False)

    return df

if __name__ == '__main__':
    df = run(path='data/raw',
        filename='recipes.csv',
        model='sentence-transformers/all-MiniLM-L6-v2')