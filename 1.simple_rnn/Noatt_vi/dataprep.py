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

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

class Lang:
    """
    a class to store word2index, index2word and word2count
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:'PAD', 1:'SOS', 2:'EOS', 3:'UNK'}
        self.n_words = 4
    
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
            
def readLangs(lang1, lang2):
    """
    read data from file and turn it into pairs
    @lang1: input language file
    @lang2: output language file
    @output: empty class lang1, empty class lang2, pairs
    """
    if lang1[-2:] == 'zh' or lang2[-2:] == 'zh':
        fname = '../iwslt-zh-en-processed/'
    else:
        fname = '../iwslt-vi-en-processed/'
        
    input_data = open(fname+lang1, encoding='utf-8').read().split('\n')
    output_data = open(fname+lang2, encoding='utf-8').read().split('\n')
    
    pairs = list(zip(input_data,output_data))
    return Lang(lang1[-2:]), Lang(lang2[-2:]), pairs

def prepareData(lang1,lang2):
    """
    sentences to pairs
    @lang1: input language file
    @lang2: output language file
    @output: well-defined class lang1&lang2, pairs
    """
    input_lang, output_lang, pairs = readLangs(lang1,lang2)
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs

def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def indexesFromSentence(lang,sentence):
    ret = []
    for word in sentence.split(' '):
        if word not in lang.word2index:
            ret.append(UNK_token)
        else:
            ret.append(lang.word2index[word] )
    return ret

def indexesFromPair(input_lang, target_lang, pair):
    input_indexes = indexesFromSentence(input_lang, pair[0])
    target_indexes = indexesFromSentence(target_lang, pair[1])
    return (input_indexes,target_indexes)

def indexesFromPairs(input_lang,target_lang,pairs):
    return [indexesFromPair(input_lang,target_lang,pair) for pair in pairs]

def tensorsFromPairsSorted(input_lang,target_lang,max_input_length,max_target_length,pairs):
    input_seqs = []
    target_seqs = []
    for i in range(len(pairs)): 
        pair = pairs[i]
        input_seqs.append(pair[0])
        target_seqs.append(pair[1])
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)
    input_lengths = [len(s) for s in input_seqs]
    target_lengths = [len(s) for s in target_seqs]
    input_padded = [pad_seq(s, max_input_length) for s in input_seqs]
    target_padded = [pad_seq(s, max_target_length) for s in target_seqs]
    input_tensor = torch.tensor(input_padded)
    target_tensor = torch.tensor(target_padded)
    
    return (input_tensor,input_lengths,target_tensor,target_lengths)

