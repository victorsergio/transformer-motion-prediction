import torch.nn as nn
import torch, math
from icecream import ic
import time
import positional_encoder as pe
"""
The architecture is based on the paper “Attention Is All You Need”. 
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017.
"""

class Transformer(nn.Module):
    # d_model : number of features
    def __init__(self, 
        n_encoder_layers: int=6,  #6
        dropout=0, 
        input_size = 3, 
        dim_val: int=512, #512
        dropout_encoder: float=0.2,
        dropout_pos_enc: float=0.1,
        n_heads: int=8,
        num_predicted_features: int=3
        ):
        
        
        super(Transformer, self).__init__()
        
        self.encoder_input_layer = nn.Linear(in_features=input_size, out_features=dim_val)
        self.positional_encoding_layer = pe.PositionalEncoder(d_model=dim_val, dropout=dropout_pos_enc)
        
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_val, nhead=n_heads, dropout=dropout_encoder)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_encoder_layers)
        
#        self.decoder = nn.Linear(feature_size,1)
        
        self.linear_mapping = nn.Linear(in_features=dim_val, out_features=num_predicted_features)
        
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.linear_mapping.bias.data.zero_()
        self.linear_mapping.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, device):
    
        src = self.encoder_input_layer(src)
        src = self.positional_encoding_layer(src)
        
       # src = self.encoder( # src shape: [batch_size, enc_seq_len, dim_val]
        #    src=src
       #     )
            
        mask = self._generate_square_subsequent_mask(len(src)).to(device)
      #  print("mask shape:", mask.size(), "src shape:", src.size())
        output = self.encoder(src, mask)
        
        #output = self.transformer_encoder(src,mask)
        output = self.linear_mapping(output)
        return output
        

