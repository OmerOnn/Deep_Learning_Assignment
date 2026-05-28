import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter


def load_and_clean_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Critical Error: CSV file not found at {csv_path}")

    try:
        df = pd.read_csv(csv_path, header=None)
        df = df.iloc[:, :3]
        df.columns = ['artist', 'song_title', 'lyrics']
        return df
    except Exception as e:
        raise RuntimeError(f"Critical Error: Failed to read or parse CSV file. Details: {str(e)}")


def clean_lyrics_text(text):
    """
    Cleans raw lyrics text before vocabulary creation and dataset construction.

    Main goals:
    1. Remove '&' tokens that appeared in the generated lyrics.
    2. Remove structural tags such as [chorus], [verse], etc.
    3. Normalize punctuation and spacing.
    4. Keep apostrophes so words like don't, I'm, you're remain meaningful.
    """
    text = str(text).lower()

    # Remove section labels such as [chorus], [verse 1], [bridge], etc.
    text = re.sub(r"\[[^\]]*\]", " ", text)

    # Convert ampersand separators into a special line-break token
    text = text.replace("&", " <line> ")

    # Normalize common apostrophe-like characters
    text = text.replace("`", "'")
    text = text.replace("’", "'")
    text = text.replace("‘", "'")

    # Keep letters, numbers, apostrophes, spaces, and angle brackets for <line>
    text = re.sub(r"[^a-zA-Z0-9'<>\s]", " ", text)

    # Remove standalone apostrophes
    text = re.sub(r"\s+'\s+", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_lyrics(text):
    """
    Converts raw lyrics text into clean tokens.
    The original '&' separator is converted into a special <LINE> token.
    """
    cleaned_text = clean_lyrics_text(text)

    if cleaned_text == "":
        return []

    tokens = cleaned_text.split()
    tokens = ["<LINE>" if token == "<line>" else token for token in tokens]

    return tokens


def build_vocab(df):
    try:
        all_words = []

        for lyric in df['lyrics'].dropna():
            tokens = tokenize_lyrics(lyric)
            all_words.extend(tokens)

        word_counts = Counter(all_words)

        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>', '<LINE>']
        vocab_words = [
            word for word, count in word_counts.items()
            if word not in special_tokens
        ]
        vocab = special_tokens + vocab_words

        word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        idx_to_word = {idx: word for idx, word in enumerate(vocab)}

        return word_to_idx, idx_to_word

    except Exception as e:
        raise RuntimeError(f"Critical Error: Failed to build vocabulary dictionaries. Details: {str(e)}")


class LyricsOnlyDataset(Dataset):
    def __init__(self, df, word_to_idx, max_len=20):
        self.df = df
        self.word_to_idx = word_to_idx
        self.max_len = max_len
        self.dataset_samples = self._process_samples()

    def _process_samples(self):
        samples = []

        for idx, row in self.df.iterrows():
            words = tokenize_lyrics(row['lyrics'])

            if len(words) == 0:
                continue

            word_indices = (
                [self.word_to_idx['<SOS>']]
                + [self.word_to_idx.get(w, self.word_to_idx['<UNK>']) for w in words]
                + [self.word_to_idx['<EOS>']]
            )

            # Create sliding chunks
            for i in range(1, len(word_indices), self.max_len):
                chunk = word_indices[i - 1: i + self.max_len]

                if len(chunk) < 2:
                    continue

                input_seq = chunk[:-1]
                target_seq = chunk[1:]

                samples.append({
                    'input_seq': torch.tensor(input_seq, dtype=torch.long),
                    'target_seq': torch.tensor(target_seq, dtype=torch.long)
                })

        return samples

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, idx):
        return self.dataset_samples[idx]


def collate_fn(batch):
    try:
        inputs = [item['input_seq'] for item in batch]
        targets = [item['target_seq'] for item in batch]

        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            inputs,
            batch_first=True,
            padding_value=0
        )

        padded_targets = torch.nn.utils.rnn.pad_sequence(
            targets,
            batch_first=True,
            padding_value=0
        )

        return padded_inputs, padded_targets

    except Exception as e:
        raise RuntimeError(f"Critical Error: Merging samples into batch failed. Details: {str(e)}")


def prepare_baseline_pipeline(csv_path, max_len=20):
    df = load_and_clean_csv(csv_path)
    word_to_idx, idx_to_word = build_vocab(df)
    dataset = LyricsOnlyDataset(df, word_to_idx, max_len=max_len)

    print("\nFinished preprocessing Baseline")
    print(f"Vocabulary size: {len(word_to_idx)}")
    print(f"Dataset samples: {len(dataset)}")

    return dataset, word_to_idx, idx_to_word