import torch
import torch.nn as nn
from transformer_model_parallel import Transformer
from efficientnet_v2 import EfficientNetV2


class Img2smiles_net(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, trg_sos_idx, enc_voc_size, dec_voc_size, d_model, n_heads, max_len,
                ffn_hidden, n_layers, drop_prob, img_embedding_dim, device):
        super().__init__()
        self.img_encoder = EfficientNetV2('b3',
                            in_channels=3,
                            pretrained=False)
        
        self.transformer_decoder = Transformer(src_pad_idx, # src에서 padding idx
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
                                  img_embedding_dim)
    
    def forward(self, img, smiles, mask):
        src = self.img_encoder.forward(img)
        out = self.transformer_decoder(src, smiles, mask)
        return out