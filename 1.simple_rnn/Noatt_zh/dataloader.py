import random
import numpy as np
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

class langDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        dataset[i] = [(input_sentence, target_sentence),(input_len, target_len)]
        """
        pair = self.pairs[key]
        length = (len(pair[0]),len(pair[1]))
        return [pair, length]

def collate_fn_zh(batch):
    """
    return (pair_batch, len_batch)
    """
    pairs = [sample[0] for sample in batch]
    input_lengths = [sample[1][0] for sample in batch]
    target_lengths = [sample[1][1] for sample in batch]
    max_input_length = max(input_lengths)
    max_target_length = max(target_lengths)

    pairstensor = tensorsFromPairsSorted(lang_zh,lang_en_zh,max_input_length,max_target_length,pairs)
    
    return [pairstensor[0], \
           (torch.from_numpy(np.array(pairstensor[1]))), \
           pairstensor[2], \
           (torch.from_numpy(np.array(pairstensor[3])))]

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