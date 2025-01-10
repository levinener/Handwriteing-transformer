import torch
import torch.nn as nn
import torch.nn.functional as F
import math
##embed_size 就是d_modle,输入序列的嵌入维度，即transformer中每个位置的特征向量维度
##num_heads 即h,注意力头的数量，就是将输入序列拆分为多少个注意力头
##head_dim,即d_k,每个注意力头的维度，由d_k=d_modle/h得到


def scaled_dot_product_attention(Q,K,V,mask=None):
    embed_size=Q.size(-1)

    ##score=QK^T/sqrt(d_k)
    scores=torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(embed_size)

    ##提供掩码矩阵，通过设置为-inf来屏蔽不应该被关注的位置
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    ##Attention Weight=Softmax(score)
    attention_weights = F.softmax(scores, dim=-1)

    ##output=Attention weight * V
    output= torch.matmul(attention_weights,V)

    return output, attention_weights


class Attention(nn.Module):
    def __init__(self, embed_size):
        super(Attention,self).__init__()
        self.embed_size= embed_size
        self.W_Q = nn.Linear(embed_size, embed_size)
        self.W_K = nn.Linear(embed_size, embed_size)
        self.W_V = nn.Linear(embed_size, embed_size)
    def forward(self,q,k,v,mask=None):
        Q=self.W_Q(q)
        K=self.W_K(k)
        V=self.W_K(v)

        out, attention_weights=scaled_dot_product_attention(Q,K,V,mask)
        return out,attention_weights

##自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.attention=Attention(embed_size)

    def forward(self,x,mask=None):
        out,attention_weights = self.attention(x,x,x,mask)
        return out,attention_weights


##交叉注意力机制
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.attention=Attention(embed_size)
        ##q来自解码器，kv来自编码器

    def forward(self,q,kv,mask=None):
        out,attention_weights = self.attention(q,kv,mask)
        return out,attention_weights


##多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, d_modle, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_modle = d_modle  ##嵌入向量的维度
        self.num_head = num_heads  ## 多头注意力的头数
        self.head_dim = d_modle // num_heads

        self.W_Q = nn.Linear(d_modle, d_modle)
        self.W_K = nn.Linear(d_modle, d_modle)
        self.W_V = nn.Linear(d_modle, d_modle)

        self.fc_out=nn.Linear(d_modle,d_modle)

    def forward(self, q, k, v, mask=None):
        batch_size=q.size(0)

        ##
        seq_len_q=q.size(1)
        seq_len_k = k.size(1)

        Q = self.W_Q(q).view(batch_size,seq_len_q,self.num_head, -1).transpose(1,2)
        K = self.W_Q(k).view(batch_size, seq_len_k, self.num_head, -1).transpose(1, 2)
        V = self.W_Q(v).view(batch_size, seq_len_k, self.num_head, -1).transpose(1, 2)


        scaled_attention,_=scaled_dot_product_attention(Q,K,V,mask)

        ##合并多头还原为(batch_size, seq_len_q, d_model)
        concat_out=scaled_attention.transpose(1,2).contiguous().view(batch_size,-1,self.d_modle)

        out=self.fc_out(concat_out)
        return out

class FFN()