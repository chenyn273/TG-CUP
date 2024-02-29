# -*- coding: utf-8 -*-
import config
import logging
import numpy as np

import torch
from torch.utils.data import DataLoader

import utils
from train import train, test, translate
from data_loader import MTDataset
from model import make_model, LabelSmoothing


class NoamOpt:
    """Optim wrapper that implements rate."""

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """Implement `lrate` above"""
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    """for batch_size 32, 5530 steps for one epoch, 2 epoch for warm-up"""
    return NoamOpt(model.afe_embed[0].d_model, 1, 10000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


def run():
    utils.set_logger(config.log_path)

    train_dataset = MTDataset(config.train_data_path)
    dev_dataset = MTDataset(config.dev_data_path)
    test_dataset = MTDataset(config.test_data_path)

    logging.info("-------- Dataset Build! --------")
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=config.batch_size,
                                collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=config.batch_size,
                                 collate_fn=test_dataset.collate_fn)

    logging.info("-------- Get Dataloader! --------")
    # 初始化模型
    model = make_model(config.afe_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    model_par = torch.nn.DataParallel(model)
    # 训练
    if config.use_smoothing:
        criterion = LabelSmoothing(size=config.afe_vocab_size, padding_idx=config.padding_idx, smoothing=0.1)
        criterion.cuda()
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    if config.use_noamopt:
        optimizer = get_std_opt(model)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    train(train_dataloader, dev_dataloader, model, model_par, criterion, optimizer)
    test(test_dataloader, model, criterion)


def check_opt():
    """check learning rate changes"""
    import numpy as np
    import matplotlib.pyplot as plt
    model = make_model(config.src_vocab_size, config.tgt_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    opt = get_std_opt(model)
    # Three settings of the lrate hyperparameters.
    opts = [opt,
            NoamOpt(512, 1, 20000, None),
            NoamOpt(256, 1, 10000, None)]
    plt.plot(np.arange(1, 50000), [[opt.rate(i) for opt in opts] for i in range(1, 50000)])
    plt.legend(["512:10000", "512:20000", "256:10000"])
    plt.show()


def one_sentence_translate(sent,code_edit, beam_search=True):
    # 初始化模型
    model = make_model(config.afe_vocab_size, config.n_layers,
                       config.d_model, config.d_ff, config.n_heads, config.dropout)
    BOS = utils.afe_tokenizer_load().bos_id()  # 2
    EOS = utils.afe_tokenizer_load().eos_id()  # 3
    code_edit = [code_edit]
    code_edit_tokens = []
    for x in code_edit:
        _1 = []
        _1.append(6)  # <code_edit_st>
        for y in x:
            for z in utils.afe_tokenizer_load().EncodeAsIds(utils.afe_tokenizer_load().detokenize(y))[1:]:
                _1.append(z)
        _1.append(7)  # <code_edit_ed>
        code_edit_tokens.append(_1)
    afe_tokens = [[BOS] + [4] + utils.afe_tokenizer_load().EncodeAsIds(sent) + [5] + code_edit_tokens[index] + [EOS] for
                  index, sent in
                  enumerate([sent])]
    batch_input = torch.LongTensor(np.array(afe_tokens)).to(config.device)
    translate(batch_input, model, use_beam=beam_search)



if __name__ == "__main__":
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
    import warnings
    warnings.filterwarnings('ignore')
    run()
    # translate_example()
