# models.py

import numpy as np
import torch
import torch.nn as nn
from transformer import *
import warnings
warnings.filterwarnings("ignore")

class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0/self.voc_size) * len(next_chars)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, train_text, vocab_index, dev_text=None):
        self.chunk_size = 20
        self.d_model = 500
        self.train = train_text
        self.dev = dev_text
        self.voc_size = len(vocab_index)
        self.vocab_index = vocab_index
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=10)
        self.transformer  = nn.TransformerEncoder(self.transformer_layer, 20, mask_check=True)
        self.embedding = nn.Embedding(self.voc_size, self.d_model)
        self.pos_encoder = PositionalEncoding(d_model=self.d_model, num_positions=self.chunk_size)
        self.softmax = nn.Softmax(dim=-1)

    # It takes a context and returns the log probability distribution over the next characters given that context as
    # a numpy vector of length equal to the vocabulary size.
    def get_next_char_log_probs(self, context):
        log_probs = np.zeros([self.voc_size])
        for i in range(0, len(context), self.chunk_size - 1):
            chunk = ' ' + context[i:i+self.chunk_size - 1]
            chunk_tensor = torch.tensor([self.vocab_index.index_of(c) for c in chunk[:-1]]).unsqueeze(0)  # Exclude the last character
            input = self.embedding(chunk_tensor)
            input = self.pos_encoder(input)
            mask = torch.triu(torch.ones(len(input), len(input)) * 0, diagonal=1)
            output = self.transformer.forward(input, mask=mask)
            probs = self.softmax(output)
            log_probs_chunk = np.log(probs.detach().numpy())
            target_index = self.vocab_index.index_of(chunk[-1])  # Get the index of the last character
            log_probs += log_probs_chunk[0, -1, target_index]  # Update log_probs with the log probability of the last character
        log_probs = log_probs - np.max(log_probs)  # Subtract the maximum for numerical stability
        log_probs = log_probs - np.log(np.sum(np.exp(log_probs)))  # Normalize so that the probabilities sum to 1
        return log_probs
    
    def get_log_prob_sequence(self, next_chars, context):
        log_prob_from_single_probs = 0.0
        for i in range(0, len(next_chars)):

            next_char_log_probs = self.get_next_char_log_probs(context + next_chars[0:i])
            log_prob_from_single_probs += next_char_log_probs[self.vocab_index.index_of(next_chars[i])]

        return log_prob_from_single_probs


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    NLM = NeuralLanguageModel(train_text, vocab_index)

    return NLM