# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        # print("input tensor shape", self.input_tensor.shape)
        # print("input tensor", self.input_tensor)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)
        # print("output tensor", self.output_tensor)
        # print("output tensor shape", self.output_tensor.shape)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.d_model = d_model
        self.d_internal = d_internal
        self.num_classes = num_classes
        self.num_layers = num_layers
        # self.layers = nn.ModuleList([TransformerLayer(self.d_model, self.d_internal) for _ in range(self.num_layers)])
        self.layers = [TransformerLayer(self.d_model, self.d_internal) for _ in range(self.num_layers)]
        # raise Exception("Implement me")

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        print("transformer forward")
        # (1) adding positional encodings to the input (see the PositionalEncoding class; but we recommend leaving these out for now) 
        # pos_encoding = PositionalEncoding(self.d_model, self.num_positions, True)
        # indices = pos_encoding(indices)
        
        # (2) using one or more of your TransformerLayers; 
        print("indices", indices.shape)
        for layer in self.layers:
            indices = layer.forward(indices)
            # indices, attention = layer(indices)
        
        # for i in range(0, self.num_layers):
        #     temp = TransformerLayer(self.d_model, self.d_internal)
        #     indices = temp(indices)
        
        # (3) using Linear and softmax layers to make the prediction. You are
        # simultaneously making predictions over each position in the sequence. Your network should return the
        # log probabilities at the output layer (a 20x3 matrix) as well as the attentions you compute, which are then
        # plotted for you for visualization purposes in plots
        linear = nn.Linear(self.d_model, self.num_classes)
        output = linear(indices)
        softmax = nn.Softmax(dim=1)
        output = softmax(output)
        return output
        # raise Exception("Implement me")


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        # raise Exception("Implement me")

    def forward(self, input_vecs):
        # (1) self-attention (single-headed is fine; you can use either backward-only or bidirectional attention); 
        # Wq = nn.Linear(self.d_model, self.d_model)
        # Wk = nn.Linear(self.d_model, self.d_model)
        # Wv = nn.Linear(self.d_model, self.d_model)
        # d = input_vecs.size(1)
        # d = input_vecs.shape[0]
        d = input_vecs.shape[1]
        print("d", d)
        print("input_vecs.shape", input_vecs.shape)
        Wq = nn.Parameter(torch.randn(self.d_internal, d))
        Wk = nn.Parameter(torch.randn(self.d_internal, d))
        Wv = nn.Parameter(torch.randn(d, d))
        # Wv = nn.Parameter(torch.randn(self.d_internal, d))
        
        # # # x = input_vecs.shape[0]
        # # q = torch.matmul(Wq, input_vecs[0].float())
        # # k = torch.matmul(Wk, input_vecs[0].float())
        # # v = torch.matmul(Wv, input_vecs[0].float())
        # q = torch.matmul(Wq, input_vecs.float().t())
        # k = torch.matmul(Wk, input_vecs.float().t())
        # v = torch.matmul(Wv, input_vecs.float().t())
        # # keys = Wk.matmul(input_vecs.float().T).T
        # # values = Wv.matmul(input_vecs.float().T).T
        # # softmax(QK^T/sqrt(d_k))V
        # q = q.t()
        # similarity = torch.matmul(q, k) / np.sqrt(self.d_internal)
        # # similarity = torch.matmul(q, k.transpose(-1, 0)) / np.sqrt(self.d_internal)
        # # similarity = torch.matmul(q, k.transpose(-2, -1))/np.sqrt(self.d_internal)
        # print("similarity", similarity)
        # similarity = torch.nn.functional.softmax(similarity, dim=-1)
        # output = torch.matmul(similarity, v.t())
        
        output = torch.zeros(input_vecs.size(0), self.d_internal)
        # output = torch.zeros_like(input_vecs.float().t())
        for i in range(input_vecs.size(0)):
            # q_i = torch.attematmul(Wq, input_vecs[i].float())
            q_i = torch.matmul(Wq, input_vecs[i].float())
            k_i = torch.matmul(Wk, input_vecs[i].float())
            v_i = torch.matmul(Wv, input_vecs[i].float())
            similarity_i = torch.matmul(q_i, k_i.transpose(-1, 0)) / np.sqrt(q_i.size(-1))
            # similarity_i = torch.matmul(q_i, k_i) / np.sqrt(self.d_internal)
            similarity_i = torch.nn.functional.softmax(similarity_i, dim=-1)
            # output_i = torch.matmul(similarity_i, v_i)
            output_i = similarity_i * v_i
            # output_i += input_vecs.float()
            # output_i += Wv.matmul(input_vecs[i].float())
            output_i = output_i[:20]
            output[i] = output_i
        
        # (2) residual connection; 
        # Define a weight matrix for the residual connection
        W_residual = torch.randn((input_vecs.size(1), self.d_internal), requires_grad=True)

        # Transform input_vecs to a 10000x50 matrix
        input_vecs_transformed = torch.matmul(input_vecs.float(), W_residual)

        # Add the residual connection to the output
        output += input_vecs_transformed

        
        # (3) Linear layer, nonlinearity, and Linear layer; 
        # # linear1 = nn.Linear(self.d_internal, d)
        # linear1 = nn.Linear(self.d_model, d)
        # # linear1 = nn.Linear(self.d_model, self.d_internal)
        # output = linear1(output.t())
        # Define a linear layer with the right input and output dimensions
        # linear1 = nn.Linear(input_vecs.size(1), self.d_internal)
        linear1 = nn.Linear(self.d_internal, self.d_model)

        # Apply the linear transformation to output
        output = linear1(output)
        relu = nn.ReLU()
        output = relu(output)
        # # linear2 = nn.Linear(self.d_model, self.d_internal)
        # linear2 = nn.Linear(self.d_internal, d)
        # output = linear2(output.t())
        # Define a linear layer with the right input and output dimensions
        linear2 = nn.Linear(self.d_internal, self.d_model)

        # Apply the linear transformation to output
        output = linear2(output)
        print("output.shape", output.shape)
        print("input_vecs.shape", input_vecs.shape)
        # (4) final residual connection. 
        # Define a linear layer with the right input and output dimensions
        # Define a linear layer with the right input and output dimensions
        linear1 = nn.Linear(input_vecs.size(1), self.d_model)

        # Apply the linear transformation to input_vecs
        input_vecs_transformed = linear1(input_vecs.float())

        # Add input_vecs_transformed to output
        output += input_vecs_transformed        
        # output += input_vecs
        
        # return output, similarity
        return output
        # raise Exception("Implement me")
    


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.tensor(np.asarray(range(0, input_size))).type(torch.LongTensor)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    model = Transformer(vocab_size=27, num_positions=20, d_model=100, d_internal=20, num_classes=3, num_layers=2)
    model.zero_grad()
    model.train()
    # trainOutputNP = np.array([ex.output for ex in train])
    # trainSet = LetterCountingExample(train, trainOutputNP, Indexer())
    # print("train", train)
    # print("dev", dev)
    # trainInputTensor = torch.tensor(train[i].input_tensor for i in range(len(train)))
    trainInputTensor = [train[i].input_tensor for i in range(len(train))]
    trainInputTensor = torch.stack(trainInputTensor)
    print("train input tensor", trainInputTensor)
    print("train input tensor.shape", trainInputTensor.shape)
    # trainOutputTensor = train.output_tensor
    model.forward(trainInputTensor)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print("test for param", name)
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    
    num_epochs = 10
    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        # You can use batching if you'd like
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)
        loss_fcn = nn.NLLLoss()
        for ex_idx in ex_idxs:
            loss = loss_fcn(train, 3) # TODO: Run forward and compute loss
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
