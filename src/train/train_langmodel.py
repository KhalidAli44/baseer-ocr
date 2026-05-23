import os
import json
import pickle
import re
from collections import defaultdict, Counter
from langmodel import DawgNode
from config import MODELS_DIR

CORPUS_PATH = "data/books.txt"
BIGRAM_PATH = f"{MODELS_DIR}/langmodel/bigrams.json"
DAWG_PATH = f"{MODELS_DIR}/langmodel/dawg.pkl"
SMOOTHING = 0.01
MIN_WORD_FREQUENCY = 2  # Only include words appearing at least this many times


def _load_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().lower()


def _build_bigrams(text):
    chars = [c for c in text if c.isalpha() or c == " "]
    counts = defaultdict(lambda: defaultdict(float))
    totals = defaultdict(float)

    for a, b in zip(chars, chars[1:]):
        if a == " " or b == " ":
            continue
        counts[a][b] += 1.0
        totals[a] += 1.0

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    probs = {}
    for a in alphabet:
        probs[a] = {}
        total = totals[a] + SMOOTHING * len(alphabet)
        for b in alphabet:
            probs[a][b] = (counts[a][b] + SMOOTHING) / total

    return probs


def _build_dawg_from_word_frequencies(word_frequencies, min_freq=2):
    """Build DAWG from words that occur at least min_freq times"""
    root = DawgNode()
    
    # Filter words by frequency
    filtered_words = [word for word, count in word_frequencies.items() if count >= min_freq]
    
    print(f"  Words after frequency filter (≥{min_freq}): {len(filtered_words):,}")
    
    for word in filtered_words:
        node = root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = DawgNode()
            node = node.children[ch]
        node.is_end = True

    return root


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(CORPUS_PATH):
        print(f"Corpus not found at {CORPUS_PATH}")
        print("Download a plain text corpus e.g. from gutenberg.org and save it there.")
        return

    print("Loading corpus...")
    text = _load_corpus(CORPUS_PATH)
    
    # Count word frequencies
    all_words = re.findall(r"[a-z]+", text)
    word_frequencies = Counter(all_words)
    
    total_unique_words = len(word_frequencies)
    total_words = len(all_words)
    
    print(f"  {len(text):,} characters")
    print(f"  {total_words:,} total words")
    print(f"  {total_unique_words:,} unique words")
    
    # Show distribution info
    words_once = sum(1 for count in word_frequencies.values() if count == 1)
    print(f"  {words_once:,} words occur only once")
    print(f"  {total_unique_words - words_once:,} words occur 2+ times")

    print("Building bigram table...")
    bigrams = _build_bigrams(text)
    with open(BIGRAM_PATH, "w", encoding="utf-8") as f:
        json.dump(bigrams, f)
    print(f"  Saved: {BIGRAM_PATH}")

    print("Building DAWG...")
    dawg = _build_dawg_from_word_frequencies(word_frequencies, MIN_WORD_FREQUENCY)
    with open(DAWG_PATH, "wb") as f:
        pickle.dump(dawg, f)
    print(f"  Saved: {DAWG_PATH}")


if __name__ == "__main__":
    train()