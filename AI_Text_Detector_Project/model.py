import torch
import torch.nn as nn

class AITextDetector(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128):
        super().__init__()
        # Embedding turns integers into dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        
        # LSTM processes the context of the sentence
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Final classification layer
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        # Summarizer: We take the output of the final word in the sequence
        summary = lstm_out[:, -1, :]
        
        return self.sigmoid(self.fc(summary))
