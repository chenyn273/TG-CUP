import config
from data_loader import subsequent_mask

import math
import copy
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_encoder import GNNEncoder

DEVICE = config.device


class LabelSmoothing(nn.Module):
    """Implement label smoothing."""

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        # Embedding层
        self.lut = nn.Embedding(vocab, d_model)
        # Embedding维数
        self.d_model = d_model

    def forward(self, x):
        # 返回x对应的embedding矩阵（需要乘以math.sqrt(d_model)）
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)


        pe = torch.zeros(max_len, d_model, device=DEVICE)

        position = torch.arange(0., max_len, device=DEVICE).unsqueeze(1)

        div_term = torch.exp(torch.arange(0., d_model, 2, device=DEVICE) * -(math.log(10000.0) / d_model))


        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)


        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def attention(query, key, value, mask=None, dropout=None):

    d_k = query.size(-1)


    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)


    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)


    p_attn = F.softmax(scores, dim=-1)


    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert d_model % h == 0

        self.d_k = d_model // h

        self.h = h

        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)

        return self.linears[-1](x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()

        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self, x):

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)


        return self.a_2 * (x - mean) / torch.sqrt(std ** 2 + self.eps) + self.b_2


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):

        return x + self.dropout(sublayer(self.norm(x)))


def clones(module, N):

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Encoder(nn.Module):
    # layer = EncoderLayer
    # N = 6
    def __init__(self, layer, N):
        super(Encoder, self).__init__()

        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):

        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        # d_model
        self.size = size

    def forward(self, x, mask):

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 复制N个encoder layer
        self.layers = clones(layer, N)
        # Layer Norm
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, node_states, src_mask, node_mask, tgt_mask):

        for layer in self.layers:
            x = layer(x, memory, node_states, src_mask, node_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, gnn_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        # Self-Attention
        self.self_attn = self_attn

        self.src_attn = src_attn
        self.gnn_attn = gnn_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 4)

    def forward(self, x, memory, node_states, src_mask, node_mask, tgt_mask):

        m = memory


        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # Node-Attention
        x = self.sublayer[2](x, lambda x: self.gnn_attn(x, node_states, node_states, node_mask))
        return self.sublayer[3](x, self.feed_forward)


class Transformer(nn.Module):
    def __init__(self, afe_encoder, gnn_encoder, decoder, afe_embed, generator):
        super(Transformer, self).__init__()

        ''' ------------ afe -------------'''
        self.afe_encoder = afe_encoder
        self.gnn_encoder = gnn_encoder
        self.decoder = decoder
        '''包含code node action和注释代码嵌入'''
        self.afe_embed = afe_embed
        self.generator = generator

    def encode(self, src, src_mask, node_values, node_len, node_as_output, edge_pret2ch, edge_prev2next,
               edge_align,
               edge_com2sub):
        node_embeddings = self.afe_embed(node_values)
        node_state, node_mask, graph_state = self.gnn_encoder(node_embeddings, node_len, node_as_output, edge_pret2ch,
                                                              edge_prev2next,
                                                              edge_align, edge_com2sub)
        return self.afe_encoder(self.afe_embed(src), src_mask), node_state, node_mask

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        memory, node_states, node_mask = encoder_output[0], encoder_output[1], encoder_output[2]
        return self.decoder(self.afe_embed(tgt), memory, node_states, src_mask, node_mask, tgt_mask)

    def forward(self, new_cmt_mask, node_values, node_len, node_as_output, edge_pret2ch, edge_prev2next, edge_align,
                edge_com2sub, afe_src, new_cmt, afe_mask):
        ''' ----------- afe cup --------------'''

        return self.decode(
            self.encode(afe_src, afe_mask, node_values, node_len, node_as_output, edge_pret2ch,
                        edge_prev2next,
                        edge_align, edge_com2sub), afe_mask, new_cmt, new_cmt_mask)


class Generator(nn.Module):
    # vocab: tgt_vocab
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()

        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):

        return F.log_softmax(self.proj(x), dim=-1)


def make_model(afe_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy

    attn = MultiHeadedAttention(h, d_model).to(DEVICE)

    ff = PositionwiseFeedForward(d_model, d_ff, dropout).to(DEVICE)

    position = PositionalEncoding(d_model, dropout).to(DEVICE)

    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        GNNEncoder(d_model, 4),
        # None,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(attn), c(ff), dropout).to(DEVICE), N).to(DEVICE),
        nn.Sequential(Embeddings(d_model, afe_vocab).to(DEVICE), c(position)),
        Generator(d_model, afe_vocab)).to(DEVICE)


    for p in model.parameters():
        if p.dim() > 1:

            nn.init.xavier_uniform_(p)
    return model.to(DEVICE)


def batch_greedy_decode(model, src, src_mask, node_values, node_len, node_as_output, edge_pret2ch, edge_prev2next,
                        edge_align,
                        edge_com2sub, max_len=64, start_symbol=2, end_symbol=3):
    batch_size, src_seq_len = src.size()
    results = [[] for _ in range(batch_size)]
    stop_flag = [False for _ in range(batch_size)]
    count = 0

    memory = model.encode(src, src_mask, node_values, node_len, node_as_output, edge_pret2ch,
                          edge_prev2next,
                          edge_align,
                          edge_com2sub)
    tgt = torch.Tensor(batch_size, 1).fill_(start_symbol).type_as(src.data)

    for s in range(max_len):
        tgt_mask = subsequent_mask(tgt.size(1)).expand(batch_size, -1, -1).type_as(src.data)
        out = model.decode(memory, src_mask, Variable(tgt), Variable(tgt_mask))

        prob = model.generator(out[:, -1, :])
        pred = torch.argmax(prob, dim=-1)

        tgt = torch.cat((tgt, pred.unsqueeze(1)), dim=1)
        pred = pred.cpu().numpy()
        for i in range(batch_size):
            # print(stop_flag[i])
            if stop_flag[i] is False:
                if pred[i] == end_symbol:
                    count += 1
                    stop_flag[i] = True
                else:
                    results[i].append(pred[i].item())
            if count == batch_size:
                break

    return results
