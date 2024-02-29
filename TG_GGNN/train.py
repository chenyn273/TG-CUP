import torch
import torch.nn as nn
from torch.autograd import Variable

import logging
import sacrebleu
from tqdm import tqdm

import config
from beam_decoder import beam_search
from model import batch_greedy_decode
from utils import afe_tokenizer_load


def run_epoch(data, model, loss_compute):
    total_tokens = 0.
    total_loss = 0.

    for batch in tqdm(data):
        out = model(batch.new_cmt_mask,
                    batch.node_value, batch.node_len, batch.node_as_output, batch.edge_prt2ch, batch.edge_prev2next,
                    batch.edge_align, batch.edge_com2sub, batch.afe_encoder_input, batch.new_cmt,
                    batch.afe_encoder_mask)
        loss = loss_compute(out, batch.new_cmt_y, batch.afe_ntokens)

        total_loss += loss
        total_tokens += batch.afe_ntokens
    return total_loss / total_tokens


def train(train_data, dev_data, model, model_par, criterion, optimizer):
    """训练并保存模型"""
    # 初始化模型在dev集上的最优Loss为一个较大值
    best_bleu_score = 0.0
    best_loss = 10000000000.0
    early_stop = config.early_stop
    for epoch in range(1, config.epoch_num + 1):
        # 模型训练
        model.train()
        # train_loss = run_epoch(train_data, model_par,
        #                        MultiGPULossCompute(model.generator, criterion, config.device_id, optimizer))
        train_loss = run_epoch(train_data, model_par,
                               LossCompute(model.generator, criterion, optimizer))
        logging.info("Epoch: {}, loss: {}".format(epoch, train_loss))
        # 模型验证
        model.eval()
        # dev_loss = run_epoch(dev_data, model_par,
        #                      MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        dev_loss = run_epoch(dev_data, model_par,
                             LossCompute(model.generator, criterion, None))
        bleu_score = evaluate(dev_data, model)
        logging.info('Epoch: {},  Bleu Score: {}'.format(epoch, bleu_score))

        # 如果当前epoch的模型在dev集上的loss优于之前记录的最优loss则保存当前模型，并更新最优loss值
        if best_bleu_score < bleu_score or best_loss > dev_loss:
            if best_bleu_score < bleu_score:
                best_bleu_score = bleu_score
            if best_loss > dev_loss:
                best_loss = dev_loss
            torch.save(model.state_dict(), config.model_path)
            early_stop = config.early_stop
            logging.info("-------- Save Best Model! --------")

        else:
            early_stop -= 1
            logging.info("Early Stop Left: {}".format(early_stop))
        if early_stop == 0:
            logging.info("-------- Early Stop! --------")
            break


class LossCompute:
    """简单的计算损失和进行参数反向传播更新训练的函数"""

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return loss.data.item() * norm.float()


class MultiGPULossCompute:
    """A multi-gpu loss compute and train function."""

    def __init__(self, generator, criterion, devices, opt=None, chunk_size=5):
        # Send out to different gpus.
        self.generator = generator
        self.criterion = nn.parallel.replicate(criterion, devices=devices)
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, out, targets, normalize):
        total = 0.0
        generator = nn.parallel.replicate(self.generator, devices=self.devices)
        out_scatter = nn.parallel.scatter(out, target_gpus=self.devices)
        out_grad = [[] for _ in out_scatter]
        targets = nn.parallel.scatter(targets, target_gpus=self.devices)

        # Divide generating into chunks.
        chunk_size = self.chunk_size
        for i in range(0, out_scatter[0].size(1), chunk_size):
            # Predict distributions
            out_column = [[Variable(o[:, i:i + chunk_size].data,
                                    requires_grad=self.opt is not None)]
                          for o in out_scatter]
            gen = nn.parallel.parallel_apply(generator, out_column)

            # Compute loss.
            y = [(g.contiguous().view(-1, g.size(-1)),
                  t[:, i:i + chunk_size].contiguous().view(-1))
                 for g, t in zip(gen, targets)]
            loss = nn.parallel.parallel_apply(self.criterion, y)

            # Sum and normalize loss
            l_ = nn.parallel.gather(loss, target_device=self.devices[0])
            l_ = l_.sum() / normalize
            total += l_.data

            # Backprop loss to output of transformer
            if self.opt is not None:
                l_.backward()
                for j, l in enumerate(loss):
                    out_grad[j].append(out_column[j][0].grad.data.clone())

        # Backprop all loss through transformer.
        if self.opt is not None:
            out_grad = [Variable(torch.cat(og, dim=1)) for og in out_grad]
            o1 = out
            o2 = nn.parallel.gather(out_grad,
                                    target_device=self.devices[0])
            o1.backward(gradient=o2)
            self.opt.step()
            if config.use_noamopt:
                self.opt.optimizer.zero_grad()
            else:
                self.opt.zero_grad()
        return total * normalize


def evaluate(data, model, mode='dev', use_beam=True):
    """在data上用训练好的模型进行预测，打印模型翻译结果"""
    sp_afe = afe_tokenizer_load()
    trg = []
    res = []
    with torch.no_grad():
        # 在data的英文数据长度上遍历下标
        for batch in tqdm(data):
            # 对应的中文句子
            cn_sent = batch.new_cmt_text
            src = batch.afe_encoder_input
            src_mask = (src != 0).unsqueeze(-2)
            if use_beam:
                decode_result, _ = beam_search(model, src, src_mask, batch.node_value, batch.node_len,
                                               batch.node_as_output,
                                               batch.edge_prt2ch, batch.edge_prev2next,
                                               batch.edge_align, batch.edge_com2sub, config.max_len,
                                               config.padding_idx, config.bos_idx, config.eos_idx,
                                               config.beam_size, config.device)
            else:
                decode_result = batch_greedy_decode(model, src, src_mask,
                                                    max_len=config.max_len)
            decode_result = [h[0] for h in decode_result]
            translation = [sp_afe.decode_ids(_s) for _s in decode_result]
            trg.extend(cn_sent)
            res.extend(translation)
    if mode == 'test':
        with open(config.output_path, "w") as fp:
            for i in range(len(trg)):
                fp.write('========== trg =========== \n')
                fp.write(trg[i] + '\n')
                fp.write('========== res =========== \n')
                fp.write(res[i] + '\n')

    trg = [trg]
    bleu = sacrebleu.corpus_bleu(res, trg)
    return float(bleu.score)


def test(data, model, criterion):
    with torch.no_grad():
        # 加载模型
        model.load_state_dict(torch.load(config.model_path))
        model_par = torch.nn.DataParallel(model)
        model.eval()
        # 开始预测
        # test_loss = run_epoch(data, model_par,
        #                       MultiGPULossCompute(model.generator, criterion, config.device_id, None))
        # test_loss = run_epoch(data, model_par,
        #                        LossCompute(model.generator, criterion, None))
        bleu_score = evaluate(data, model, 'test')
        logging.info('Bleu Score: {}'.format(bleu_score))


def translate(src, model, use_beam=True):
    """用训练好的模型进行预测单句，打印模型翻译结果"""
    sp_afe = afe_tokenizer_load()
    with torch.no_grad():
        model.load_state_dict(torch.load(config.model_path))
        model.eval()
        src_mask = (src != 0).unsqueeze(-2)
        if use_beam:
            decode_result, _ = beam_search(model, src, src_mask, config.max_len,
                                           config.padding_idx, config.bos_idx, config.eos_idx,
                                           config.beam_size, config.device)
            decode_result = [h[0] for h in decode_result]
        else:
            decode_result = batch_greedy_decode(model, src, src_mask, max_len=config.max_len)
        translation = [sp_afe.decode_ids(_s) for _s in decode_result]
        print(translation[0])
