import os
import pickle
import pandas as pd
import sklearn
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def get_dataset(data, set, regenerate= False, target='channel'):

    output = 'data/dataset/embedder'

    if regenerate:
        df = generate_dataset(data, set, 3, target)
        train_set, test_set = train_test_split(df, test_size=0.1, random_state=42)
        train_set.to_csv(os.path.join(output, f'train_{set}_{target}.csv'), index=False)
        test_set.to_csv(os.path.join(output, f'test_{set}_{target}.csv'), index=False)
    else:
        train_set = pd.read_csv(os.path.join(output, f'train_{set}_{target}.csv'))
        test_set = os.path.join(output, f'test_{set}_{target}.csv')

    return train_set, test_set

def generate_dataset(data, set, k, target):
    global model
    if set not in {'trigger', 'action'}:
        raise ValueError("Invalid value for 'set'. Must be 'trigger' or 'action'.")

    prev = 'IF ' if set == 'trigger' else 'THEN '

    channel_column = f'{set}ChannelTitle'
    function_column = f'{set}Title'
    data.dropna(subset=[channel_column, function_column], inplace=True)

    df = pd.DataFrame(columns=['input', 'positive', 'negative'])

    if target == 'both' or target == 'function':
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        pickle_file = 'data/dataset/embedder/embeddings.pkl'
        if os.path.exists(pickle_file):
            print(f"Pickle file '{pickle_file}' found. Loading embeddings...")
            with open(pickle_file, "rb") as f:
                embeddings = pickle.load(f)
            print("Embeddings loaded from the pickle file.")
        else:
            print(f"Pickle file '{pickle_file}' not found. Generating embeddings...")
            embeddings = []
            for idx, k_row in tqdm(data.iterrows(), total=len(data)):
                embeddings.append(model.encode(str(k_row[function_column])))

            with open(pickle_file, "wb") as f:
                pickle.dump(embeddings, f)
            print("Embeddings have been saved to 'embeddings.pkl'.")

    for idx, k_row in tqdm(data.iterrows(), total=len(data)):
        target_channel = k_row[channel_column]
        target_function = k_row[function_column]
        if target == 'channel':
            positive = data[
                (data[channel_column] == target_channel)
                &
                (data[function_column] != target_function)
                ]

            negative = data[
                (data[channel_column] != target_channel)
                &
                (data[function_column] != target_function)
                ]

        elif target == 'function':
            target_function_embedding = embeddings[idx]
            similarities = []
            for i in range(len(embeddings)):
                if i != idx:
                    similarities.append(model.similarity(target_function_embedding, embeddings[i]))
            threshold = 0.8
            # Get the index of the most similar function
            # Most similar are those above 0.8
            most_similar_idx = [i for i, score in enumerate(similarities) if score > threshold and score != 1]
            positive = data.iloc[most_similar_idx]
            negative = data[~data.index.isin(most_similar_idx)]
        try:
            positive_samples_same_channel = positive.sample(n=k, replace=False, random_state=42)
            positive_samples_same_channel = positive_samples_same_channel[[channel_column, function_column]]
        except ValueError as e:
            print(f"Not enough positive samples for row {idx}: {e}")
            continue

        try:
            negative_samples_same_channel = negative.sample(n=2 * k, replace=False, random_state=42)
            negative_samples_same_channel = negative_samples_same_channel[[channel_column, function_column]]
        except ValueError as e:
            print(f"Not enough negative samples for row {idx}: {e}")
            continue

        input = str(prev + target_channel + " " + target_function)

        def get_samples(rows, prefix):
            return [f"{prefix}{row[channel_column]} {row[function_column]}" for _, row in rows.iterrows()]

        positives = (
            get_samples(positive_samples_same_channel, prev)
        )
        negatives = (
            get_samples(negative_samples_same_channel, prev)
        )
        row = pd.DataFrame({
            "input": input,
            "positive": [positives],
            "negative": [negatives]
        })

        df = pd.concat([df, row], axis=0, ignore_index=True)

    return df


if __name__ == '__main__':
    sets = {'trigger', 'action'}
    data = pd.read_csv('data/processed/original_scored.csv')
    train_set, test_set = get_dataset(data, 'action', regenerate=True, target='function')
    """
    for set in sets:
        train_set, test_set = get_dataset(data, 'action', regenerate=True, target='function')
    """