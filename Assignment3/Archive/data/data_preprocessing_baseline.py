import os
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

def build_vocab(df):
    try:
        all_words = []
        for lyric in df['lyrics'].dropna():
            tokens = lyric.lower().split()
            all_words.extend(tokens)
            
        word_counts = Counter(all_words)
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        vocab = special_tokens + [word for word, count in word_counts.items()]
        
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
            lyric_text = str(row['lyrics'])
            words = lyric_text.lower().split()
            
            # Encapsulate with sequence markers
            word_indices = [self.word_to_idx['<SOS>']] + [self.word_to_idx.get(w, self.word_to_idx['<UNK>']) for w in words] + [self.word_to_idx['<EOS>']]
            
            # Create sliding windows
            for i in range(1, len(word_indices), self.max_len):
                chunk = word_indices[i-1 : i + self.max_len]
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
        
        padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
        padded_targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0)
        
        return padded_inputs, padded_targets
    except Exception as e:
        raise RuntimeError(f"Critical Error: Merging samples into batch failed. Details: {str(e)}")

def prepare_baseline_pipeline(csv_path, max_len=20):
    df = load_and_clean_csv(csv_path)
    word_to_idx, idx_to_word = build_vocab(df)
    dataset = LyricsOnlyDataset(df, word_to_idx, max_len=max_len)
    print("\nFinished preprocessing Baseline")
    return dataset, word_to_idx, idx_to_word