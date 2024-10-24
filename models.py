
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, RobertaModel
from transformers import logging
from sklearn.cross_decomposition import CCA
from soft_dtw_cuda import SoftDTW


logging.set_verbosity_warning()

class Q_Encoder(nn.Module):

    def __init__(self):
    
        super().__init__()
        
        self.weights = 'FacebookAI/roberta-base'
        self.model = RobertaModel.from_pretrained(self.weights)
        # self.weights = 'bert-base-uncased'
        # self.model = BertModel.from_pretrained(self.weights)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        

    def forward(self, input_ids, attn_mask):

        self.model.eval()
        with torch.no_grad():
            x = self.model(input_ids, attn_mask)['last_hidden_state'][:, 0, :]
        return x


class CrossAttentionBlock(nn.Module):

    def __init__(self, embed_dim, n_head, n_hidden):
        
        super().__init__()
        self.n_head = n_head
        self.n_hidden = n_hidden
        self.norm_first = False
        self.relu = nn.ReLU()
        self.attn = nn.MultiheadAttention(embed_dim, n_head, batch_first=True, dropout=0.5)
        self.linear1 = nn.Linear(embed_dim, n_hidden)
        self.dropout = nn.Dropout(0.5)
        self.linear2 = nn.Linear(n_hidden, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def attn_block(self, q, k, v):
        return self.attn(q, k, v)[0]

    def add_norm_block(self, x, y):
        return self.norm(x + y)
    
    def ff_block(self, x):
        return self.relu(self.linear2(self.linear1(x)))

    def forward(self, q, k, v):
            
        # Post-LN Transformer
        # f_a = self.attn_block(q, k, v)
        # f_b = self.add_norm_block(q, f_a)
        # fc_1 = self.ff_block(f_b)
        # f_n = self.add_norm_block(f_b, fc_1)

        # Pre-LN Transformer
        x = self.norm(q)
        f_a = self.attn_block(x, k, v)
        f_b = self.add_norm_block(q, f_a)
        fc_1 = self.ff_block(f_b)
        f_n = self.add_norm_block(fc_1, q+f_a)

        return f_n


"""
Use this to reduce down the spatial information of visual features
and construct separate MLP for a and v
"""
class Pool(nn.Module):

    def __init__(self, type):
        super().__init__()
        self.type = type
    
    def forward(self, x):

        if self.type == 'max':
            x, _ = torch.max(x, dim=-1)
        elif self.type == 'mean':
            x = torch.mean(x, dim=-1)
        return x

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, output_dim),
        )
    
    def forward(self, x):

        batch_size, num_frames, num_features = x.shape
        x = x.view(-1, num_features)  # Flatten frames for batch processing
        x = self.model(x)
        x = x.view(batch_size, num_frames, -1)  # Reshape back to sequence format
        return x
    
class VisualProj(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            MLP(output_dim, output_dim)              # (B, 1, output_dim)
        )

    def forward(self, x):
        
        return self.model(x)

class AudioProj(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_dim),        # (B, 60, output_dim)
            MLP(output_dim, output_dim)              # (B, 1, output_dim)
        )
    def forward(self, x):
        
        return self.model(x)
        

class self_attention(nn.Module):

    def __init__(self):
        super().__init__()
        
    def forward(self, q, k, v):

        out = F.scaled_dot_product_attention(q, k, v)
        out = F.dropout(F.relu(out))

        return out


class pred_block(nn.Sequential):
    def __init__(self, embed_dim, n_ans):
        super().__init__(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(embed_dim, n_ans),
        )

        
class AVQA(nn.Module):

    def __init__(self):

        super().__init__()
        
        embed_dim = 512
        n_head = 4
        n_hidden = 512
        n_ans = 42
        
        self.a_mlp = MLP(128, embed_dim)
        self.v_mlp = MLP(512, embed_dim)
        self.q_mlp = MLP(768, embed_dim)
        
        self.a_proj = AudioProj(embed_dim)
        self.v_proj = VisualProj(embed_dim)
        self.a_pool_mlp = MLP(128, 128)
        self.v_pool_mlp = MLP(128, 128)
        self.av_pool_mlp = MLP(128, 128)
        self.avq_pool_mlp = MLP(512, 256)

        self.q_encoder = Q_Encoder()
        # self.av_loss = SoftDTW(use_cuda=True, gamma=0.1)

        self.vq_attn = CrossAttentionBlock(embed_dim, n_head, n_hidden)
        self.av_attn = CrossAttentionBlock(embed_dim, n_head, n_hidden)
        self.aq_attn = CrossAttentionBlock(embed_dim, n_head, n_hidden)
        self.avq_attn = CrossAttentionBlock(embed_dim, n_head, n_hidden)

        self.pred_block = pred_block(256, n_ans)
        # self.noise = GaussianNoise()
        # self._initialize_weights()
    
    def ff_block(self, x):
        return self.avq_linear2(self.activation(self.avq_linear1(x)))

    """
    Apply alignment loss between audio and visual representation
    """
    def av_align_loss(self, a, v):

        margin = 0.5
        cos_sim = F.cosine_similarity(a, v, dim=-1)  # Shape: (batch_size, sequence_length)
        sim_loss = (margin - cos_sim).mean()

        return sim_loss
    
    """
    Apply loss for disagreement on the sectors where each modality does better
    """
    def agreement_loss(self, aq, vq):

        a_distr = F.softmax(aq, dim=-1)
        v_distr = F.softmax(vq, dim=-1)
        div_loss = F.relu(torch.sum(a_distr * (torch.log(a_distr) - torch.log(v_distr)), dim=-1)).mean()
        
        return div_loss
        
    
    def forward(self, a, v, q, attn_mask):  

        q_raw = self.q_encoder(q, attn_mask).unsqueeze(1)

        a = self.a_mlp(a)                           # (B, 60, 512)
        v = self.v_mlp(v)                           # (B, 60, 512)
        q = self.q_mlp(q_raw)                       # (B, 1, 128)
        
        
        vq = self.vq_attn(q, v, v)                  # (B, 1, 512)
        aq = self.aq_attn(q, a, v)                  # (B, 1, 512)

        aqvq = self.avq_pool_mlp(F.adaptive_max_pool1d(torch.concat((aq, vq), dim=-1), 512))    # (B, 1, 512)
        

        av_loss = self.av_align_loss(a, v)
        ag_loss = self.agreement_loss(aq, vq)
        
        pred = self.pred_block(aqvq).squeeze(1)

        return av_loss, ag_loss, pred
    
