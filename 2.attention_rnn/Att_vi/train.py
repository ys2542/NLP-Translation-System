import random
from scipy import stats
import re
import unicodedata
import string
import time
import datetime
import math
import socket
hostname = socket.gethostname()

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy

from dataprep import tensorsFromPairsSorted

import sacrebleu
import subprocess

MAX_LENGTH = 70
batch_size = 2
teacher_forcing_ratio = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()#use arange instead of range
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len).to(device)
    seq_range_expand = Variable(seq_range_expand)
#     if sequence_length.is_cuda:
#         seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length, max_length=MAX_LENGTH):

    length = torch.tensor(length).to(device)
    max_target_length = max(length)
    if max_target_length > max_length:
        target = target[:, 0:max_length]
    #print('target shape:', target.shape)

    logits_flat = logits.view(-1, logits.size(-1)).to(device)
    #print('logits_flat shape:', logits_flat.shape)
    log_probs_flat = F.log_softmax(logits_flat).to(device)
    #print('log_probs_flat shape:', log_probs_flat.shape)
    target_flat = target.contiguous().view(-1, 1).to(device)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    losses = losses_flat.view(*target.size())
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()
    return loss

def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    encoder_outputs, encoder_hidden = encoder_outputs.to(device), encoder_hidden.to(device)
    
    decoder_input = torch.tensor([SOS_token]*batch_size).to(device)
    decoder_hidden = encoder_hidden[:decoder.n_layers].to(device) # Use last (forward) hidden state from encoder
    max_target_length = max(target_lengths)
    if max_target_length > max_length: 
        max_target_length = max_length
    all_decoder_outputs = torch.zeros(batch_size, max_target_length, decoder.output_size)
    
    loss = 0

#     for ei in range(input_length):
#         encoder_output, encoder_hidden = encoder(
#             input_tensor[ei], encoder_hidden)
#         encoder_outputs[ei] = encoder_output[0, 0]

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_output, decoder_hidden, decoder_attention = decoder_output.to(device), decoder_hidden.to(device), decoder_attention.to(device)
            all_decoder_outputs[:, di] = decoder_output
            decoder_input = target_batches[:, di] # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            ni = topi[0][0]
            all_decoder_outputs[:, di] = decoder_output
            decoder_input = topi.squeeze().detach()

            if ni == EOS_token:
                break
                
    loss = masked_cross_entropy(
                all_decoder_outputs.contiguous(),
                target_batches.contiguous(),
                target_lengths)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    #return loss.item()
    return loss.data[0]

def evaluate(input_batches, input_lengths, output_lang, target_batches, target_lengths, encoder, decoder, max_length=MAX_LENGTH):      
    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)
    
    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    encoder_outputs, encoder_hidden = encoder_outputs.to(device), encoder_hidden.to(device)

    # Create starting vectors for decoder
    eval_batch_size = input_batches.shape[0]
    decoder_input = torch.tensor([SOS_token] * eval_batch_size).to(device)
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    decoded_words = ''    
    max_target_length = max(target_lengths)
    if max_target_length > max_length: 
        max_target_length = max_length
    all_decoder_outputs = torch.zeros(eval_batch_size, max_target_length, decoder.output_size)
    
    # Run through decoder
    for di in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_output, decoder_hidden, decoder_attention = decoder_output.to(device), decoder_hidden.to(device), decoder_attention.to(device)

        # Choose top word from output
        all_decoder_outputs[:, di] = decoder_output
        topv, topi = decoder_output.data.topk(1)
        ni = topi.cpu().data[0].item()
        pi = topi.cpu().data
        for batch in range(eval_batch_size):
            ni = pi[batch].item()
            if ni == EOS_token:
                decoded_words += ('\n')
                break
            else:
                decoded_words += (' ' + output_lang.index2word[ni])
            
        decoder_input = pi.to(device)

    # Set back to training mode
    decoded_words = decoded_words[0:len(decoded_words)]
    encoder.train(True)
    decoder.train(True)
    
    return decoded_words, all_decoder_outputs

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))