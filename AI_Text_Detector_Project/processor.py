import re
import collections
import torch

class TextProcessor:
    def __init__(self, max_len=100, vocab_size=10000):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}

    def build_vocab(self, texts):
        all_words = " ".join(texts).lower().split()
        counts = collections.Counter(all_words)
        for i, (word, _) in enumerate(counts.most_common(self.vocab_size-2)):
            self.word_to_idx[word] = i + 2

    def transform(self, text):
        # Basic cleaning and conversion to numbers
        words = re.sub(r'[^\w\s]', '', str(text).lower()).split()
        tokens = [self.word_to_idx.get(w, 1) for w in words]
        
        # Ensure all sequences are the same length
        if len(tokens) < self.max_len:
            tokens += [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]
        return torch.tensor(tokens).long()
