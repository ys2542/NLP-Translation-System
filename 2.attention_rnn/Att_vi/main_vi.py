import numpy as np
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
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import sacrebleu
import subprocess

from dataprep import prepareData, indexesFromPairs, tensorsFromPairsSorted
from att_model import EncoderRNN, AttnDecoderRNN
from train import masked_cross_entropy, train, evaluate, as_minutes, time_since
from dataloader import langDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 2
MAX_LENGTH = 70
attn_model = 'general'
hidden_size = 500
n_layers = 2
dropout = 0.1

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_epochs = 5
epoch = 0

lang_vi, lang_en_vi, train_pairs_vi = prepareData('train.tok.vi','train.tok.en')
_, _, val_pairs_vi = prepareData('dev.tok.vi','dev.tok.en')
_, _, test_pairs_vi = prepareData('test.tok.vi','test.tok.en')

train_id_vi = indexesFromPairs(lang_vi, lang_en_vi, train_pairs_vi)
val_id_vi = indexesFromPairs(lang_vi, lang_en_vi, val_pairs_vi)
test_id_vi = indexesFromPairs(lang_vi, lang_en_vi, test_pairs_vi)

def collate_fn_vi(batch):
    """
    return (pair_batch, len_batch)
    """
    pairs = [sample[0] for sample in batch]
    input_lengths = [sample[1][0] for sample in batch]
    target_lengths = [sample[1][1] for sample in batch]
    max_input_length = max(input_lengths)
    max_target_length = max(target_lengths)

    pairstensor = tensorsFromPairsSorted(lang_vi,lang_en_vi,max_input_length,max_target_length,pairs)
    
    return [pairstensor[0], \
           (torch.from_numpy(np.array(pairstensor[1]))), \
           pairstensor[2], \
           (torch.from_numpy(np.array(pairstensor[3])))]

train_vi = langDataset(train_id_vi)
val_vi = langDataset(val_id_vi)
test_vi = langDataset(test_id_vi)

train_loader_vi = torch.utils.data.DataLoader(dataset=train_vi,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn_vi,
                                              shuffle=True)
val_loader_vi = torch.utils.data.DataLoader(dataset=val_vi,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn_vi,
                                              shuffle=True)
test_loader_vi = torch.utils.data.DataLoader(dataset=test_vi,
                                              batch_size=batch_size,
                                              collate_fn=collate_fn_vi,
                                              shuffle=True)


# Initialize models
encoder = EncoderRNN(lang_vi.n_words, hidden_size, n_layers, dropout=dropout).to(device)
decoder = AttnDecoderRNN(attn_model, hidden_size, lang_en_vi.n_words, n_layers, dropout=dropout).to(device)

# Initialize optimizers and criterion
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
criterion = nn.CrossEntropyLoss().to(device)

# Keep track of time elapsed and running averages
start = time.time()

def test_model_loss(loader, output_lang, encoder, decoder):
    encoder.eval()
    decoder.eval()
    total = 0
    total_loss = 0
    for i, (input_batches, input_lengths, target_batches, target_lengths) in enumerate(loader):
        input_batches, input_lengths, target_batches, target_lengths = input_batches.to(device), input_lengths.to(device), target_batches.to(device), target_lengths.to(device)
        decoded_words, all_decoder_outputs = evaluate(input_batches, input_lengths, output_lang, target_batches, target_lengths, encoder, decoder)
        #print(all_decoder_outputs.shape)
        loss = masked_cross_entropy(
                all_decoder_outputs.contiguous(),
                target_batches.contiguous(),
                target_lengths)
        total_loss += loss.item()
        total += 1
        
    return (total_loss / total)

def test_model_score(loader, output_lang, encoder, decoder, targetlang):
    encoder.eval()
    decoder.eval()
    total = 0
    total_loss = 0
    predict_file = 'predict_temp'
    predict_lines = open(predict_file, 'w')
    for i, (input_batches, input_lengths, target_batches, target_lengths) in enumerate(loader):
        input_batches, input_lengths, target_batches, target_lengths = input_batches.to(device), input_lengths.to(device), target_batches.to(device), target_lengths.to(device)
        decoded_words, all_decoder_outputs = evaluate(input_batches, input_lengths, output_lang, target_batches, target_lengths, encoder, decoder)
        loss = masked_cross_entropy(
                all_decoder_outputs.contiguous(), # -> batch x seq
                target_batches.contiguous(), # -> batch x seq
                target_lengths)
        
        total_loss += loss
        total += 1
        predict_lines.write(''.join(decoded_words) + '\n')
    predict_lines.close()
        
    if targetlang == 'zh':
        target_file = '../iwslt-zh-en-processed/dev.tok.en'
    else:
        target_file = '../iwslt-vi-en-processed/dev.tok.en'
    result = subprocess.run('cat {} | sacrebleu {}'.format(predict_file,target_file),shell=True,stdout=subprocess.PIPE)
    score = get_blue_score(str(result))
        
    return (total_loss / total), score

train_loss_step = []
val_loss_step = []
print_every = 100
plot_every = 1000
n_iters = len(train_loader_vi)

while epoch < n_epochs:
    epoch += 1
    plot_losses = []
    print_loss_avg = 0
    plot_loss_total = 0
    # Get training data for this cycle
    for i, (input_batches, input_lengths, target_batches, target_lengths) in enumerate(train_loader_vi):
        input_batches, input_lengths, target_batches, target_lengths = input_batches.to(device), input_lengths.to(device), target_batches.to(device), target_lengths.to(device)
        loss = train(
            input_batches, input_lengths, target_batches, target_lengths,
            encoder, decoder,
            encoder_optimizer, decoder_optimizer, criterion
        )
        plot_loss_total += loss.item()
        print_loss_avg += loss.item()
        
        if i > 0 and i % print_every == 0:
            print_loss_avg = plot_loss_total / print_every
            plot_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, i / n_iters),
                                         i, i / n_iters * 100, print_loss_avg))

        if i > 0 and i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            train_loss_step.append(loss)
            val_l = test_model_loss(val_loader_vi, lang_en_vi, encoder, decoder)
            val_loss_step.append(val_l)
            print('Epoch: [{}/{}], Step: [{}/{}], Train Loss: {}, Validation Loss: {}'.format(
                       epoch+1, n_epochs, i+1, len(train_loader_vi), loss, val_l))
            
torch.save(encoder.state_dict(), "att_encoder_vi")
torch.save(decoder.state_dict(), "att_decoder_vi")
        
out = {"train_loss_1000_step": train_loss_step, 
       "validation_loss_1000_step": val_loss_step       }

with open('att_output_vi.txt', 'w') as file:
     file.write(json.dumps(out))

