import os

import pandas as pd

from libs.preprocessing.utils import compute_score, save_dataframe, save_full_dataset

pd.set_option('display.max_colwidth', None)


def run(path: str, filename: str, model: str, output: str = None) -> pd.DataFrame:
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
    cleaned_df = pd.DataFrame(columns=['desc', 'recipe', 'trigger', 'action'])
    # Add to the cleaned_df desc column the concat of the column title and desc of df with a space in between
    cleaned_df['desc'] = df['title'] + '. ' + df['desc']
    cleaned_df['recipe'] = ('IF ' + df['triggerChannelTitle'] + ' ' + df['triggerTitle'] + ' THEN '
                            + df['actionChannelTitle'] + ' ' + df['actionTitle'])
    cleaned_df['trigger'] = 'TRIGGER SERVICE: ' + df['triggerChannelTitle'] + ', TRIGGER EVENT: ' + df['triggerTitle']
    cleaned_df['action'] = 'ACTION SERVICE: ' + df['actionChannelTitle'] + ', ACTION EVENT: ' + df['actionTitle']
    # add recipegen column that follows this format "Flickr <sep> Flickr.New_public_photos <sep> Twitter <sep> Twitter.Post_a_tweet"
    cleaned_df['recipegen'] = (df["triggerChannelTitle"].str.replace(" ", "_")
                               + ' <sep> ' +
                               df["triggerTitle"].str.replace(" ", "_")
                               + ' <sep> ' +
                               df["actionChannelTitle"].str.replace(" ", "_")
                               + ' <sep> ' +
                               df["actionTitle"].str.replace(" ", "_"))

    cleaned_df.dropna(inplace=True)
    cleaned_df.reset_index(drop=True, inplace=True)

    cleaned_df = compute_score(cleaned_df, model)
    cleaned_df.to_csv(os.path.join(output, 'scored.csv'), index=False)

    df['score'] = cleaned_df['score']
    df = df[df['score'] >= 0.4].reset_index(drop=True)
    df.to_csv(os.path.join(output, 'original_scored.csv'), index=False)

    if output is not None:
        cleaned_df.to_csv(os.path.join(output, 'scored.csv'), index=False)

        gold_df = cleaned_df[cleaned_df['score'] >= 0.4].reset_index(drop=True)

        save_full_dataset(gold_df, output)

        data_specs = [
            ('recipe', ['desc', 'recipe', 'score'], "From the description of a rule: identify the 'trigger', "
                                                    "identify the 'action', write a IF 'trigger' THEN 'action' rule."),
            ('trigger', ['desc', 'trigger', 'score'], "From the description of a rule: identify the 'trigger'"),
            ('action', ['desc', 'action', 'score'], "From the description of a rule: identify the 'action'"),
            ('recipegen', ['desc', 'recipegen', 'score'], "Recipegen++ dataset")
        ]

        for filename, columns, instruction in data_specs:
            save_dataframe(gold_df, output, filename, columns, instruction)



    return cleaned_df


if __name__ == '__main__':

    run(path='data/raw',
        filename='recipes.csv',
        model='sentence-transformers/all-MiniLM-L6-v2',
        output='data/processed/')
