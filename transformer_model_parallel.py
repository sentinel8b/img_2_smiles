import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def make_pad_mask(k):
    mask = ~k.ne(0).unsqueeze(1).unsqueeze(2)
    return mask

def make_no_peak_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal = 1).type(torch.BoolTensor)
    return mask

def create_mask(q):
    size = q.size(1)
    pad_mask = make_pad_mask(q)
    no_peak_mask = make_no_peak_mask(size)
    
    return ~torch.maximum(pad_mask , no_peak_mask)


## Scaled dot attention
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(0.1)
    def forward(self, q, k ,v, mask = None, e = -1e9):
        batch_size, head, length, d_tensor = k.size()

        k_t = k.view(batch_size, head, d_tensor, length)
        score = (q @ k_t)/math.sqrt(d_tensor)
        if mask is not None:
            score = score.masked_fill(mask == 0, e) # 마스크 존재하면 마스크 원소가 0이면 1e-12값으로 변경

        score = self.softmax(score)
        score = self.dropout(score)
        v = score @ v

        return v

## Multi-Head Attention

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def split_head(self, tensor): #실제로 split하는게 아니라 shape을 변경해줌
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model//self.n_head
        tensor = tensor.view(batch_size, self.n_head, length, d_tensor)
        return tensor

    def concat_head(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        tensor = tensor.view(batch_size, length, head * d_tensor)
        return tensor

    def forward(self, q,k,v, mask = None):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        q = self.split_head(q)
        k = self.split_head(k)
        v = self.split_head(v)

        out = self.attention(q,k,v, mask = mask)

        out = self.concat_head(out)
        out = self.w_concat(out)

        return out

## FFN

class PositionwiseFeedForward(nn.Module):
    
    def __init__(self,d_model,hidden,drop_prob = 0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.linear1 = nn.Linear(d_model,hidden)
        self.linear2 = nn.Linear(hidden,d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = drop_prob)
    
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


## Encoder Layer

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(d_model, n_head)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p = drop_prob)

        self.ffn = PositionwiseFeedForward(d_model = d_model, hidden = ffn_hidden, drop_prob = drop_prob)

        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p = drop_prob)

    def forward(self, x):
        attn_output = self.attention(q = x, k = x, v = x)
        attn_output = self.dropout1(attn_output)
        out1 = self.norm1(x + attn_output)
        # x = self.dropout1(x)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.norm2(out1 + ffn_output)
        # x = self.dropout2(x)

        return out2

## Encoder

class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.embed = nn.Linear(1536, d_model)
        self.dropout = nn.Dropout(p = drop_prob)
        self.layers = nn.ModuleList([EncoderLayer(d_model = d_model, ffn_hidden = ffn_hidden, n_head = n_head, drop_prob = drop_prob) for _ in range(n_layers)])
        
    def positional_encoding(self, x,  max_len, d_model):
        encoding = torch.zeros(max_len, d_model, device = x.device) # (len, d_model) 사이즈의 0 행렬을 만들어줌
        encoding.requires_grad = False #학습안되게 False

        pos = torch.arange(0, max_len) # 0,1,2, .... , max_len-1까지의 row vector
        pos = pos.float().unsqueeze(dim = 1) #float으로 바꾸고 차원 추가

        _2i = torch.arange(0, d_model, step = 2).float() #짝수인 row vector 만들어줌

        encoding[:, 0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        encoding[:, 1::2] = torch.cos(pos/(10000**(_2i/d_model)))
        batch_size, seq_len, _ = x.size()

        return encoding[:seq_len, :]
        
    def forward(self, x):
        x = self.embed(x)
        pe = self.positional_encoding(x, 100, self.d_model)
        x = x + pe
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)

        return x


## Decoder Layer


class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model = d_model, n_head = n_head)

        self.norm1 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(p = drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model = d_model, n_head = n_head)

        self.norm2 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(p = drop_prob)

        self.ffn = PositionwiseFeedForward(d_model = d_model, hidden= ffn_hidden, drop_prob= drop_prob)

        self.norm3 = nn.LayerNorm(d_model)

        self.dropout3 = nn.Dropout(p = drop_prob)

    def forward(self, dec, enc, trg_mask):
        attn1 = self.self_attention(q = dec, k = dec, v = dec, mask = trg_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.norm1(attn1 + dec)
        
        attn2 = self.enc_dec_attention(q = out1, k = enc, v = enc)
        attn2 = self.dropout2(attn2)
        out2 = self.norm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        
        out3 = self.norm3(ffn_output + out2)
        # x = self.dropout3(x)

        return out3


                                   

## Decoder Layer

class Decoder(nn.Module):
    def __init__(self,dec_voc_size,max_len,d_model,ffn_hidden,n_head,n_layers,
                drop_prob):
        super().__init__()
        
        #Embedding
        self.embed = nn.Embedding(num_embeddings = dec_voc_size, embedding_dim = d_model)
        
        #Add decoder layers
        self.layers = nn.ModuleList([DecoderLayer(d_model = d_model,
                                                 ffn_hidden = ffn_hidden,
                                                 n_head = n_head,
                                                 drop_prob = drop_prob) for _ in range(n_layers)])
        self.max_len = max_len
        #Linear
        self.linear = nn.Linear(d_model,dec_voc_size)
        self.softmax = nn.Softmax()
        self.d_model = d_model
        
    def positional_encoding(self, x,  max_len, d_model):
        encoding = torch.zeros(max_len, d_model, device = x.device) # (len, d_model) 사이즈의 0 행렬을 만들어줌
        encoding.requires_grad = False #학습안되게 False

        pos = torch.arange(0, max_len) # 0,1,2, .... , max_len-1까지의 row vector
        pos = pos.float().unsqueeze(dim = 1) #float으로 바꾸고 차원 추가

        _2i = torch.arange(0, d_model, step = 2).float() #짝수인 row vector 만들어줌

        encoding[:, 0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        encoding[:, 1::2] = torch.cos(pos/(10000**(_2i/d_model)))
        batch_size, seq_len, _ = x.size()

        return encoding[:seq_len, :]
    
    def forward(self,trg,src,trg_mask):
        #Compute Embedding
        trg = self.embed(trg)
        trg *= math.sqrt(self.d_model)
        #Get Positional Encoding
        trg_pe = self.positional_encoding(trg, self.max_len, self.d_model)
        
        #Embedding + Positional Encoding
        trg = trg + trg_pe
        
        #Compute Decoder layers
        for layer in self.layers:
            trg = layer(trg,src,trg_mask)
        
        #pass to LM head
        output = self.linear(trg)

        return output
    
class Transformer(nn.Module):
    
    def __init__(self,src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                ffn_hidden, n_layers, drop_prob, img_embedding_dim):
        super().__init__()
        #Get <PAD> idx
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.dropout = nn.Dropout(p = 0.1)
        #Encoder
        self.encoder = Encoder(enc_voc_size = enc_voc_size,
                              max_len = max_len,
                              d_model = d_model,
                              ffn_hidden = ffn_hidden,
                              n_head = n_head,
                              n_layers = n_layers,
                              drop_prob = drop_prob)
        
        #Decoder
        self.decoder = Decoder(dec_voc_size = dec_voc_size,
                              max_len = max_len,
                              d_model = d_model,
                              ffn_hidden = ffn_hidden,
                              n_head = n_head,
                              n_layers = n_layers,
                              drop_prob = drop_prob)
     
    def forward(self,src,trg, mask):
        #Get Mask
        src = self.dropout(src)
        #Compute Encoder
        enc_src = self.encoder(src)
        #Compute Decoder
        output = self.decoder(trg, enc_src, mask)
        
        return output

