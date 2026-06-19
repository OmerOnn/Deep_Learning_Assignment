import os
import re
import pandas as pd
import pretty_midi
import torch
from torch.utils.data import Dataset
from collections import Counter
import warnings

warnings.filterwarnings("ignore", module="pretty_midi")


def load_and_clean_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Critical Error: CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path, header=None)
    df = df.iloc[:, :3]
    df.columns = ['artist', 'song_title', 'lyrics']

    return df


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


def extract_midi_features(midi_path):
    if not midi_path or not os.path.exists(midi_path):
        return torch.zeros(12)

    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        melody_inst = None

        for inst in pm.instruments:
            if inst.name.lower() == "melody":
                melody_inst = inst
                break

        if melody_inst is None and len(pm.instruments) > 0:
            for inst in pm.instruments:
                if not inst.is_drum:
                    melody_inst = inst
                    break

        if melody_inst is None:
            return torch.zeros(12)

        pitch_counts = [0.0] * 12

        for note in melody_inst.notes:
            pitch_class = note.pitch % 12
            pitch_counts[pitch_class] += (note.end - note.start)

        total_duration = sum(pitch_counts)

        if total_duration > 0:
            pitch_counts = [count / total_duration for count in pitch_counts]

        return torch.tensor(pitch_counts, dtype=torch.float32)

    except Exception:
        return torch.zeros(12)


def find_matching_midi(song_title, midi_files, midi_dir):
    """
    Finds the MIDI file matching the given song title.
    """
    clean_title = str(song_title).lower().strip()
    clean_title_underscore = clean_title.replace(" ", "_")

    for midi_file in midi_files:
        midi_name = midi_file.lower()
        midi_name_no_ext = midi_name.replace(".mid", "").replace(".midi", "")
        midi_name_spaces = midi_name_no_ext.replace("_", " ")

        if clean_title_underscore in midi_name:
            return os.path.join(midi_dir, midi_file)

        if clean_title in midi_name_spaces:
            return os.path.join(midi_dir, midi_file)

        if midi_name_spaces in clean_title:
            return os.path.join(midi_dir, midi_file)

    return None


class LyricsMelodyDataset(Dataset):
    def __init__(self, df, word_to_idx, midi_dir, max_len=20):
        self.df = df
        self.word_to_idx = word_to_idx
        self.midi_dir = midi_dir
        self.max_len = max_len
        self.dataset_samples = self._process_samples()

    def _process_samples(self):
        samples = []
        midi_files = os.listdir(self.midi_dir) if os.path.exists(self.midi_dir) else []

        for idx, row in self.df.iterrows():
            song_title = str(row['song_title'])
            matched_midi = find_matching_midi(song_title, midi_files, self.midi_dir)

            if matched_midi:
                melody_vector = extract_midi_features(matched_midi)
            else:
                melody_vector = torch.zeros(12)

            words = tokenize_lyrics(row['lyrics'])

            if len(words) == 0:
                continue

            word_indices = (
                [self.word_to_idx['<SOS>']]
                + [self.word_to_idx.get(w, self.word_to_idx['<UNK>']) for w in words]
                + [self.word_to_idx['<EOS>']]
            )

            for i in range(1, len(word_indices), self.max_len):
                chunk = word_indices[i - 1: i + self.max_len]

                if len(chunk) < 2:
                    continue

                input_seq = chunk[:-1]
                target_seq = chunk[1:]

                samples.append({
                    'input_seq': torch.tensor(input_seq, dtype=torch.long),
                    'target_seq': torch.tensor(target_seq, dtype=torch.long),
                    'melody': melody_vector
                })

        return samples

    def __len__(self):
        return len(self.dataset_samples)

    def __getitem__(self, idx):
        return self.dataset_samples[idx]


def collate_fn_melody(batch):
    inputs = [item['input_seq'] for item in batch]
    targets = [item['target_seq'] for item in batch]
    melodies = torch.stack([item['melody'] for item in batch])

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

    return padded_inputs, padded_targets, melodies


def prepare_melody_pipeline(csv_path, midi_dir, max_len=20):
    df = load_and_clean_csv(csv_path)
    word_to_idx, idx_to_word = build_vocab(df)
    dataset = LyricsMelodyDataset(df, word_to_idx, midi_dir, max_len=max_len)

    print("\nFinished preprocessing Melody")
    print(f"Vocabulary size: {len(word_to_idx)}")
    print(f"Dataset samples: {len(dataset)}")

    return dataset, word_to_idx, idx_to_word