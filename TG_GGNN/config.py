import torch

d_model = 512
n_heads = 8
n_layers = 6
d_k = 64
d_v = 64
d_ff = 2048
dropout = 0.1
padding_idx = 0
bos_idx = 2
eos_idx = 3

afe_vocab_size = 32000
batch_size = 8
epoch_num = 200
early_stop = 5
lr = 0.001

# greed decode的最大句子长度
max_len = 100
# beam size for bleu
beam_size = 5
# Label Smoothing
use_smoothing = False
# NoamOpt
use_noamopt = True

data_dir = './data'
train_data_path = './data/json/train_clean_BPE.jsonl'
dev_data_path = './data/json/valid_clean_BPE.jsonl'
test_data_path = './data/json/test_clean_BPE.jsonl'
model_path = './experiment/model.pth'
log_path = './experiment/train.log'
output_path = './experiment/output.txt'

# gpu_id and device id is the relative id
# thus, if you wanna use os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
# you should set CUDA_VISIBLE_DEVICES = 2 as main -> gpu_id = '0', device_id = [0, 1]
gpu_id = '0'
device_id = [0, 1, 2, 3]

# set device
if gpu_id != '':
    device = torch.device(f"cuda:{gpu_id}")
else:
    device = torch.device('cpu')
