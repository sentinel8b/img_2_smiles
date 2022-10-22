# Basic Config 
raid_dir = '/raid/qlql323/img2smiles/'
train_size = 1000000
# Model config
img_embedding_dim = 232
d_model = 512
ffn_hidden = 2048
n_layers = 4
n_heads = 8
src_pad_idx = 0
trg_pad_idx = 0
trg_sos_idx = 8
enc_voc_size = 232
dec_voc_size = 30
max_len = 46
# max_len = 43
drop_prob = 0.1
from_ckpt = 0



# IMPORT
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import math
import tensorflow as tf
import torchvision
from torchvision import datasets, models, transforms
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from torchvision.models import EfficientNet_B3_Weights

import transformer_model_parallel
from transformer_model_parallel import Transformer
# from custom_dataset import img2selfie_dataset
from custom_dataset_parallel import img2selfie_dataset
from backbone_model import ImgEncoder
from Optim_amp import NoamOpt
from model import Img2smiles_net

from tqdm import tqdm
import numpy as np
import pandas as pd
import selfies as sf
import os
import tensorflow as tf
from PIL import Image
import psutil
import gc
import time
import wandb
import pickle
""
print("Import complete")
""

if 'device' not in globals():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read Data
train_data = pd.read_csv(raid_dir + '1mil_train.csv')

train_dataset = train_data[:train_size]
val_dataset = train_data[train_size:]

# Tokenization
with open('selfie_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
tokenizer.word_index["pad"] = 0
tokenizer.index_word[0] = "pad"

tokenizer.filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'

#train용 데이터 벡터
temp = ['<' + item + '>' for item in list(train_dataset['SELFIES'])]
train_seqs = tokenizer.texts_to_sequences(temp)
train_cap_vec = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

#val용 데이터벡터
temp = ['<' + item + '>' for item in list(val_dataset['SELFIES'])]
val_seqs = tokenizer.texts_to_sequences(temp)
val_cap_vec = tf.keras.preprocessing.sequence.pad_sequences(val_seqs, padding='post')


""
print("data prepared")
""
# Prepare training
train_path = 'train/'
train_data_set = img2selfie_dataset(raid_dir + train_path, train_dataset, train_cap_vec)
train_loader = DataLoader(train_data_set, batch_size = 512, shuffle = True, num_workers = 16, pin_memory = True)
val_data_set = img2selfie_dataset(raid_dir + train_path, val_dataset, val_cap_vec)
val_loader = DataLoader(val_data_set, batch_size = 512, shuffle = False, num_workers = 16, pin_memory = True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

torch.cuda.manual_seed_all(629)

Net = Img2smiles_net(src_pad_idx, # src에서 padding idx
                                  trg_pad_idx, # trg에서 padding idx
                                  trg_sos_idx, # trg에서 sos(start of sentence) idx
                                  enc_voc_size, # encoder vocab size -> 인코더 입력 차원
                                  dec_voc_size, # decoder vocab size -> 디코더 출력 차원
                                  d_model, # 모델 dim
                                  n_heads, # attention head 갯수
                                  max_len, # 최대길이
                                  ffn_hidden,
                                  n_layers,
                                  drop_prob,
                                  img_embedding_dim,
                                  device)
Net = torch.nn.DataParallel(Net)
Net.cuda()

# optimizer = torch.optim.AdamW(transformer_decoder.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

custom_opt = NoamOpt(d_model, 1, 4000,
            torch.optim.Adam(Net.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

# criterion = nn.CrossEntropyLoss(reduction = 'none', ignore_index=0)
criterion = nn.CrossEntropyLoss(ignore_index=0)
# criterion = DataParallelCriterion(criterion)

""
print("finished preparing training")
""

def convert_2_selfie(input_tensor):
    result = []
    for sample in input_tensor:
        tmp = []
        for pred in sample:
            char = "["+tokenizer.index_word[int(pred)]+"]"
            tmp.append(char)
        result.append("".join(tmp).replace("[<]", "").replace("[>]", "").replace("[pad]", ""))
    return result
            

# Train function
def train_one_epoch(epoch, Net, trainloader, validloader, criterion, optimizer):  
    Net.train()
    epoch_loss = 0
    valid_epoch_loss = 0
    for i, data in enumerate(tqdm(trainloader)):
        img, smiles = data
        optimizer.optimizer.zero_grad()
        smiles_batch_input = smiles[:, :-1]
        smiles_batch_target = smiles[:, 1:]
        mask = transformer_model_parallel.create_mask(smiles_batch_target)
        out = Net(img, smiles_batch_input, mask)
        if i % 2000 == 0:
            images = wandb.Image(img[:5])
            pred = convert_2_selfie(torch.argmax(out[:5], axis = -1))
            ans = convert_2_selfie(smiles_batch_target[:5])
            train_table.add_data(epoch, i//100, images, pred, ans)
        batch_loss = criterion(torch.transpose(out, -2, -1), smiles_batch_target.to(device))
        epoch_loss += batch_loss.item()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(Net.parameters(), 10)
        optimizer.step()

        # scaler.scale(batch_loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        

    with torch.no_grad():
        Net.eval()
        for i, data in enumerate(tqdm(validloader)):
            img, smiles = data
            smiles_batch_input = smiles[:, :-1]
            smiles_batch_target = smiles[:, 1:]
            mask = transformer_model_parallel.create_mask(smiles_batch_target)
            out = Net(img, smiles_batch_input, mask)
            if i % 20 == 0:
                images = wandb.Image(img[:5])
                pred = convert_2_selfie(torch.argmax(out[:5], axis = -1))
                ans = convert_2_selfie(smiles_batch_target[:5])
                
                valid_table.add_data(epoch, i//2, images, pred, ans)
            batch_loss = criterion(torch.transpose(out, -2, -1), smiles_batch_target.to(device))
            # batch_loss = torch.mean(batch_loss)
            valid_epoch_loss += batch_loss.item()
            
    return epoch_loss/len(trainloader), valid_epoch_loss/len(validloader)

# wandb monitoring
# wandb login 8f9018feb613413130ddac26827f20e04e2a0a62

""
print("train started")
""
wandb.init(project="img2smiles", resume = False)
train_table = wandb.Table(columns = ['epoch', 'step', 'images', 'result', 'ans'])
valid_table = wandb.Table(columns = ['epoch', 'step', 'images', 'result', 'ans'])

train_loss = []
valid_loss = []
start_epoch = 0
epochs = 40
best_loss = 1e5

if from_ckpt > 0:
    with open(f'ckpt/train_loss{from_ckpt}.pickle', 'rb') as handle:
        train_loss = pickle.load(handle)
    with open(f'ckpt/valid_loss{from_ckpt}.pickle', 'rb') as handle:
        valid_loss = pickle.load(handle)
    with open(f'ckpt/optimizer{from_ckpt}.pickle', 'rb') as handle:
        optimizer = pickle.load(handle)
    Net.load_state_dict(torch.load('best_model/best_model.pickle'))
    start_epoch = from_ckpt
    ########################################################################
    print("loaded weight and other data")
    ########################################################################
    
for epoch in range(start_epoch + 1, epochs + 1):
    print('-' * 10)
    print('Epoch {}/{}'.format(epoch, epochs))
    epoch_train_loss, epoch_valid_loss = train_one_epoch(epoch, Net, train_loader, val_loader, criterion, custom_opt)
    train_loss.append(epoch_train_loss)
    valid_loss.append(epoch_valid_loss)
    #compare loss to record best model
    if len(valid_loss) >= 1:
        if valid_loss[-1] < best_loss:
            print(f"valid loss on this {epoch} is better than previous one, saving model.....")
            torch.save(Net.state_dict(), 'best_model/best_model.pickle')
            best_loss = valid_loss[-1]
    #save every data to train from ckpt
    with open(f'ckpt/optimizer{epoch}.pickle', 'wb') as handle:
        pickle.dump(custom_opt, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'ckpt/train_loss{epoch}.pickle', 'wb') as handle:
        pickle.dump(train_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(f'ckpt/valid_loss{epoch}.pickle', 'wb') as handle:
        pickle.dump(valid_loss, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #save model
    torch.save(Net.state_dict(), f'ckpt/model_{epoch}.pickle')
    
    #Log on wandb monitor
    wandb.log({'train_loss': train_loss[-1], 'valid_loss': valid_loss[-1], 'lr' : custom_opt._rate, 'training_samples' : train_table, 'valid_sample' : valid_table})
    #print stats
    print(f'Epoch : [{epoch}] Train Loss : [{train_loss[-1]:.5f}], Valid Loss : [{valid_loss[-1]:.5f}]') 
    print('-' * 10)
print("train finished")
exit()

