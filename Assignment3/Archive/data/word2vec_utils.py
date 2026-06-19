import os
import torch
from gensim.models import KeyedVectors


def load_word2vec_model(word2vec_path):
    """
    Load pretrained Google News Word2Vec vectors.

    Expected file:
    data/word2vec/GoogleNews-vectors-negative300.bin
    """
    if not os.path.exists(word2vec_path):
        raise FileNotFoundError(
            f"Word2Vec file was not found at:\n{word2vec_path}"
        )

    print("Loading pretrained Word2Vec model...")
    word2vec_model = KeyedVectors.load_word2vec_format(
        word2vec_path,
        binary=True
    )
    print("Finished loading pretrained Word2Vec model.")

    return word2vec_model


def build_embedding_matrix(word_to_idx, word2vec_model, embedding_dim=300):
    """
    Build an embedding matrix for the project vocabulary.

    If a word exists in Word2Vec, its pretrained vector is copied.
    If a word does not exist in Word2Vec, it is initialized randomly.
    """
    vocab_size = len(word_to_idx)

    embedding_matrix = torch.empty(vocab_size, embedding_dim)
    torch.nn.init.uniform_(embedding_matrix, -0.05, 0.05)

    found_words = 0
    missing_words = 0

    for word, idx in word_to_idx.items():
        if word == "<PAD>":
            embedding_matrix[idx] = torch.zeros(embedding_dim)

        elif word in ["<UNK>", "<SOS>", "<EOS>", "<LINE>"]:
            torch.nn.init.uniform_(embedding_matrix[idx], -0.05, 0.05)

        elif word in word2vec_model:
            embedding_matrix[idx] = torch.tensor(
                word2vec_model[word],
                dtype=torch.float32
            )
            found_words += 1

        else:
            missing_words += 1

    total_regular_words = found_words + missing_words
    coverage = found_words / max(1, total_regular_words)

    print("Word2Vec embedding matrix created.")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Found in Word2Vec: {found_words}/{total_regular_words}")
    print(f"Coverage: {coverage:.2%}")
    print(f"Missing words: {missing_words}")

    return embedding_matrix