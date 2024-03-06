import jsonlines
import torch
import json
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import config
from utils import afe_tokenizer_load

DEVICE = config.device


def subsequent_mask(size):
    """Mask out subsequent positions."""
    # 设定subsequent_mask矩阵的shape
    attn_shape = (1, size, size)

    # 生成一个右上角(不含主对角线)为全1，左下角(含主对角线)为全0的subsequent_mask矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')

    # 返回一个右上角(不含主对角线)为全False，左下角(含主对角线)为全True的subsequent_mask矩阵
    return torch.from_numpy(subsequent_mask) == 0


class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, gnn_input, afe_encoder_text, new_cmt_text, afe_encoder_input, new_cmt=None, pad=0):
        self.node_value = gnn_input[0].to(DEVICE)
        # self.node_action = gnn_input[1].to(DEVICE)
        self.node_len = gnn_input[2]
        self.node_as_output = gnn_input[3]
        self.edge_prt2ch = gnn_input[4]
        self.edge_prev2next = gnn_input[5]
        self.edge_align = gnn_input[6]
        self.edge_com2sub = gnn_input[7]
        # self.afe_encoder_text = afe_encoder_text
        self.new_cmt_text = new_cmt_text

        afe_encoder_input = afe_encoder_input.to(DEVICE)
        # gnn_input = gnn_input.to(DEVICE)

        self.afe_encoder_input = afe_encoder_input
        # self.gnn_input = gnn_input
        # 对于当前输入的句子非空部分进行判断成bool序列
        # 并在seq length前面增加一维，形成维度为 1×seq length 的矩阵
        self.afe_encoder_mask = (afe_encoder_input != pad).unsqueeze(-2)

        # 如果输出目标不为空，则需要对decoder要使用到的target句子进行mask
        # if trg is not None :
        if new_cmt is not None:
            new_cmt = new_cmt.to(DEVICE)
            # decoder要用到的target输入部分
            self.new_cmt = new_cmt[:, :-1]
            # decoder训练时应预测输出的target结果
            self.new_cmt_y = new_cmt[:, 1:]
            # 将target输入部分进行attention mask
            self.new_cmt_mask = self.make_std_mask(self.new_cmt, pad)
            # 将应输出的target结果中实际的词数进行统计
            self.afe_ntokens = (self.new_cmt_y != pad).data.sum()

    # Mask掩码操作
    @staticmethod
    def make_std_mask(tgt, pad):
        """Create a mask to hide padding and future words."""
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class MTDataset(Dataset):
    def __init__(self, data_path):
        self.code_change_seqs, self.old_cmt_seqs, self.new_cmt_seqs, self.gnn_inputs = self.get_dataset(data_path,
                                                                                                        sort=True)
        self.sp_afe = afe_tokenizer_load()
        self.PAD = self.sp_afe.pad_id()  # 0
        self.BOS = self.sp_afe.bos_id()  # 2
        self.EOS = self.sp_afe.eos_id()  # 3

    @staticmethod
    def len_argsort(seq):
        """传入一系列句子数据(分好词的列表形式)，按照句子长度排序后，返回排序后原来各句子在数据中的索引下标"""
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    def get_dataset(self, data_path, sort=False):
        """---------------------- AFE-CUP ---------------------"""
        code_change_seqs = []
        old_cmt_seqs = []
        new_cmt_seqs = []
        gnn_inputs = []
        with open(data_path, 'r+', encoding='utf8') as f:
            js = list(jsonlines.Reader(f))
            for e in js:
                code_change_seqs.append([['<before>', x[0], '<after>', x[1],
                                          x[2].replace('equal', '<c_keep>').replace('delete', '<c_delete>').replace(
                                              'insert', '<c_insert>').replace('replace', '<c_replace>')] for x in
                                         e['code_change_seq']])
                old_cmt_seqs.append(e['src_desc'])
                new_cmt_seqs.append(e['dst_desc'])
                gnn_inputs.append({'node_values': e['node_values'],
                                   'node_actions': e['node_actions'],
                                   'edge_prt2ch': e['edge_prt2ch'],
                                   'edge_prev2next': e['edge_prev2next'],
                                   'edge_align': e['edge_align'],
                                   'edge_com2sub': e['edge_com2sub']})
        return code_change_seqs, old_cmt_seqs, new_cmt_seqs, gnn_inputs

    def __getitem__(self, idx):
        gnn_input = self.gnn_inputs[idx]
        old_cmt = self.old_cmt_seqs[idx]
        code_edit = self.code_change_seqs[idx]
        new_cmt = self.new_cmt_seqs[idx]
        return [gnn_input, old_cmt, code_edit, new_cmt]

    def __len__(self):
        return len(self.code_change_seqs)

    def collate_fn(self, batch):
        gnn_input = [x[0] for x in batch]
        node_values = [i['node_values'] for i in gnn_input]
        node_actions = [i['node_actions'] for i in gnn_input]
        edge_prt2ch = [i['edge_prt2ch'] for i in gnn_input]
        edge_prev2next = [i['edge_prev2next'] for i in gnn_input]
        edge_align = [i['edge_align'] for i in gnn_input]
        edge_com2sub = [i['edge_com2sub'] for i in gnn_input]
        node_lens = [len(i) for i in node_values]
        node_as_output = []
        for n in node_values:
            temp = []
            for i in n:
                if i == '<Node>':
                    temp.append(False)
                else:
                    temp.append(True)
            node_as_output.append(temp)
        node_values = [
            [self.sp_afe.EncodeAsIds(n)[-1] if len(self.sp_afe.EncodeAsIds(n)) > 0 else 31939 for n in nv] for nv in
            node_values]
        node_actions = [
            [self.sp_afe.EncodeAsIds(n)[-1] if len(self.sp_afe.EncodeAsIds(n)) > 0 else 31939 for n in nv] for nv in
            node_actions]
        old_cmt = [x[1] for x in batch]
        code_edit = [x[2] for x in batch]
        code_edit_tokens = []
        for x in code_edit:
            _1 = []
            _1.append(self.sp_afe.encode_as_ids('<code_edit_st>')[1])  # <code_edit_st>
            for y in x:
                for z in y:
                    if z != '':
                        _1.append(self.sp_afe.EncodeAsIds(z)[-1])
                    else:
                        _1.append(int(31939))
            _1.append(self.sp_afe.encode_as_ids('<code_edit_ed>')[1])  # <code_edit_ed>
            code_edit_tokens.append(_1)
        new_cmt = [x[3] for x in batch]

        afe_tokens = [[self.BOS] + [self.sp_afe.EncodeAsIds('<old_cmt_st>')[1]] + self.sp_afe.EncodeAsIds(sent) + [
            self.sp_afe.EncodeAsIds('<old_cmt_ed>')[1]] + code_edit_tokens[index] + [self.EOS] for
                      index, sent in
                      enumerate(old_cmt)]
        new_cmt_tokens = [[self.BOS] + self.sp_afe.EncodeAsIds(sent) + [self.EOS] for sent in new_cmt]

        batch_afe = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in afe_tokens],
                                 batch_first=True, padding_value=self.PAD)
        batch_new_cmt = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in new_cmt_tokens],
                                     batch_first=True, padding_value=self.PAD)
        batch_N_value = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in node_values],
                                     batch_first=True, padding_value=self.PAD)
        batch_N_action = pad_sequence([torch.LongTensor(np.array(l_)) for l_ in node_actions],
                                      batch_first=True, padding_value=self.PAD)

        return Batch([batch_N_value, batch_N_action, node_lens, node_as_output, edge_prt2ch, edge_prev2next, edge_align,
                      edge_com2sub], [self.sp_afe.decode(x) for x in afe_tokens], new_cmt, batch_afe,
                     batch_new_cmt, self.PAD)
