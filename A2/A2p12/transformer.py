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
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, batch_size):
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
        self.layers = nn.ModuleList([TransformerLayer(self.d_model, self.d_internal) for _ in range(self.num_layers)])
        # self.layers = [TransformerLayer(self.d_model, self.d_internal) for _ in range(self.num_layers)]
        self.linear = nn.Linear(20, self.num_classes)
        self.batch_size = batch_size
        # raise Exception("Implement me")

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # print("transformer forward")
        # (1) adding positional encodings to the input (see the PositionalEncoding class; but we recommend leaving these out for now) 
        indices = indices.unsqueeze(1)  # Reshape indices to (20, 1)
        indices = indices.expand(-1, 100)  # Repeat indices along the second dimension to get shape (20, 100)
        pos_encoding = PositionalEncoding(self.d_model, self.num_positions, True)
        indices = pos_encoding(indices)
        
        # (2) using one or more of your TransformerLayers; 
        print("indices", indices.shape, indices)
        for layer in self.layers:
            # for i in range(self.batch_size):
            #     if i >= indices.size(0):
            #         break
            #     indices[i] = layer(indices[i])
            indices = layer.forward(indices)
            # indices, attention = layer(indices)
        
        # for i in range(0, self.num_layers):
        #     temp = TransformerLayer(self.d_model, self.d_internal)
        #     indices = temp(indices)
        
        # (3) using Linear and softmax layers to make the prediction. You are
        # simultaneously making predictions over each position in the sequence. Your network should return the
        # log probabilities at the output layer (a 20x3 matrix) as well as the attentions you compute, which are then
        # plotted for you for visualization purposes in plots
        # linear = nn.Linear(20, self.num_classes)
        # print(indices)
        # print(indices.shape)
        # print(indices.float())

        # output = torch.zeros(self.batch_size, 20, 3)
        # for i in range(self.batch_size):
        #     if i >= indices.size(0):
        #         break
        #     output[i] = self.linear(indices[i].float())
        output = torch.zeros(20, 3)
        output = self.linear(indices.float())
        # print(output)
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
        self.linear1 = nn.Linear(self.d_internal, self.d_model)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(self.d_model, self.d_internal)
        # raise Exception("Implement me")

    def forward(self, input_vecs):
        # print("input_vecs.shape", input_vecs.shape)
        # Wq = nn.Parameter(torch.randn(self.d_internal, self.d_internal))
        # Wk = nn.Parameter(torch.randn(self.d_internal, self.d_internal))
        # Wv = nn.Parameter(torch.randn(self.d_internal, self.d_internal))
        Wq = nn.Parameter(torch.randn(self.d_model, self.d_internal))
        Wk = nn.Parameter(torch.randn(self.d_model, self.d_internal))
        Wv = nn.Parameter(torch.randn(self.d_model, self.d_internal))
        
        attentions = torch.zeros_like(input_vecs.float().t())
        output = torch.zeros_like(input_vecs.float().t())
        # attentions = torch.zeros_like(input_vecs.float().t())
        # output = torch.zeros_like(input_vecs.float().t())
        for i in range(input_vecs.size(0)):
            q_i = torch.matmul(input_vecs[0][i].float(), Wq)
            k_i = torch.matmul(input_vecs[0][i].float(), Wk)
            v_i = torch.matmul(input_vecs[0][i].float(), Wv)
            similarity_i = torch.matmul(q_i, k_i.transpose(-1, 0)) / np.sqrt(q_i.size(-1))
            similarity_i = torch.nn.functional.softmax(similarity_i, dim=-1)
            attentions[i] = similarity_i
            output_i = similarity_i * v_i
            output_i = output_i[:20]
            output[i] = output_i

        # attentions = torch.zeros(20, self.d_model)
        # output = torch.zeros(20, self.d_model)
        # print("input_vecs", input_vecs)
        # q_i = torch.matmul(input_vecs[0].float(), Wq)
        # k_i = torch.matmul(input_vecs[0].float(), Wk)
        # v_i = torch.matmul(input_vecs[0].float(), Wv)
        # similarity_i = torch.matmul(q_i, k_i.transpose(-1, 0)) / np.sqrt(q_i.size(-1))
        # similarity_i = torch.nn.functional.softmax(similarity_i, dim=-1)
        # output = similarity_i * v_i
        # output = output[:100]

        # (2) residual connection; 
        output += input_vecs.float()

        # (3) Linear layer, nonlinearity, and Linear layer; 
        output = self.linear1(output)
        output = self.relu(output)
        output = self.linear2(output)
        # linear1 = nn.Linear(self.d_internal, self.d_model)
        # relu = nn.ReLU()
        # linear2 = nn.Linear(self.d_model, self.d_internal)
    
        # (4) final residual connection.       
        output += input_vecs.float()
        print("attentions", attentions.shape, attentions)
        
        return output, attentions
    


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
    batch_size = 20
    num_batches = len(train) // batch_size
    model = Transformer(vocab_size=27, num_positions=20, d_model=100, d_internal=20, num_classes=3, num_layers=2, batch_size=batch_size)
    model.zero_grad()
    model.train()
    
    # trainInputTensor = [train[i].input_tensor for i in range(len(train))]
    # trainInputTensor = torch.stack(trainInputTensor)
    # for i in range(0, len(train), batch_size):
    # # Get the inputs for this batch
    #     inputs = trainInputTensor[i:i+batch_size]

    #     # Run the model on the inputs
    #     model.forward(inputs)
        
    # model.forward(trainInputTensor)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print("test for param", name)
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
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
            ex = train[ex_idx]
            (log_probs, attn_maps) = model.forward(ex.input_tensor)
            
            outputs = model.forward(train[ex_idx].input_tensor)
            print("ex_idx", ex_idx, "outputs", outputs.shape, outputs)
            # targets = torch.stack([train[ex_idx].output_tensor])
            targets = train[ex_idx].output_tensor
            
            loss = loss_fcn(outputs, targets) # TODO: Run forward and compute loss
            model.zero_grad()
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()
        
    #     for batch_idx in range(num_batches):
    #         batch_start = batch_idx * batch_size
    #         batch_end = (batch_idx + 1) * batch_size
    #         batch_indices = ex_idxs[batch_start:batch_end]

    #         # Get the inputs and targets for this batch
    #         inputs = torch.stack([train[idx].input_tensor for idx in batch_indices])
    #         # print("inputs", inputs.shape, inputs)
    #         targets = torch.stack([train[idx].output_tensor for idx in batch_indices])
    #         # print("targets", targets.shape, targets)

    #         # Run the model on the inputs and compute the loss
    #         outputs = model.forward(inputs)
    #         # print("outputs", outputs.shape, outputs)
    #         # for i in range(len(inputs)):
    #         #     # Compute the loss
    #         #     # outputs = model.forward(inputs[i])
    #         #     loss = loss_fcn(outputs[i], targets[i])
    #         loss = loss_fcn(outputs.view(-1, outputs.shape[-1]), targets.view(-1))

    #         # Backpropagation
    #         model.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         # Accumulate the loss
    #         loss_this_epoch += loss.item()

    # #         loss = loss_fcn(outputs, targets)

    # #         # Backpropagation
    # #         model.zero_grad()
    # #         loss.backward()
    # #         optimizer.step()

    # #         # Accumulate the loss
    # #         loss_this_epoch += loss.item()
    #     # for ex_idx in ex_idxs:
    #     #     # Compute the loss
    #     #     loss = loss_fcn(...)
    #     #     model.zero_grad()
    #     #     loss.backward()
    #     #     optimizer.step()
    #     #     loss_this_epoch += loss.item()
    # # print("Loss for epoch %i: %f" % (t, loss_this_epoch))
    model.eval()
    # print("model", model)
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
