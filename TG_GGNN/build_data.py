import json

import jsonlines

from process_ast.process_ast import get_ast_diff_json
from cup_utils.comment import JavadocDescPreprocessor, CommentCleaner
from cup_utils.edit import construct_diff_sequence
import sentencepiece as spm


def get_code_change_seq(old_method, new_method):

    sp = spm.SentencePieceProcessor()
    sp.Load("tokenizer/code.model")
    old_method = sp.EncodeAsPieces(old_method)
    old_method = [sp.detokenize(x) for x in old_method]
    new_method = sp.EncodeAsPieces(new_method)
    new_method = [sp.detokenize(x) for x in new_method]
    diffs = construct_diff_sequence(old_method, new_method)
    return diffs




# 输入一个样本的新旧代码
# 输出该样本GNN的输入
def build_gnn_input(old_method, new_method):
    ast_diff = get_ast_diff_json(old_method, new_method)
    # 构造 边、节点， 其中只考虑有实际含义的节点，并用bpe进行分词构造子节点
    # 返回节点列表，节点的行为列表，边的列表
    node_values = []
    node_actions = []
    edge_prt2ch = []
    edge_prev2next = []
    edge_align = []
    edge_com2sub = []
    # 暂存可以被BPE拆分的节点
    _com_nodes = []
    for index, node in enumerate(ast_diff):
        # 将node_value添加到数组
        if node['value'] == node['attribute']:
            # 跳过这类节点 用特殊标志代替
            node_values.append('<Node>')
        else:
            # 检查是否可用BPE拆分
            sp = spm.SentencePieceProcessor()
            sp.Load("tokenizer/code.model")
            res = sp.EncodeAsPieces(node['value'].lower())
            if len(res) > 1:
                _com_nodes.append(index)
                node_values.append('<Node>')
            else:
                if len(res) == 0:
                    node_values.append('<Node>')
                else:
                    node_values.append(node['value'].lower())
        # 将除了node_value以外的值添加到数组
        node_actions.append('Node' + str(node['action_type']))
        for i in node['children_ids']:
            edge_prt2ch.append([index, i])
        for i in node['next_sibling_ids']:
            edge_prev2next.append([index, i])
        for i in node['aligned_neighbor_ids']:
            if [index, i] not in edge_align and [i, index] not in edge_align:
                edge_align.append([index, i])
    # 处理可以被BPE拆分的节点
    for index in _com_nodes:
        node = ast_diff[index]
        res = sp.EncodeAsPieces(node['value'].lower())
        for j, sub_word in enumerate(res):
            node_values.append(sp.detokenize(sub_word))
            node_actions.append('Node' + str(node['action_type']))
            edge_com2sub.append([index, len(node_values) - 1])
    return node_values, node_actions, edge_prt2ch, edge_prev2next, edge_align, edge_com2sub



# 将heb-cup清洗过后的数据集转换成BPE Transformer形式的数据集
def construct_data():
    files = ['test_clean', 'valid_clean', 'train_clean']
    cmt_cleaner = CommentCleaner(True)
    for file in files:
        with open('data/json/' + file + '.jsonl', 'r+', encoding='utf8') as f:
            js = list(jsonlines.Reader(f))
            for i, e in enumerate(js):
                # 删除AST没有发生改变的 无效样本
                if str(e).lower().__contains__('ggfs') or str(e).lower().__contains__('igfs'):
                    continue
                e['code_change_seq'] = get_code_change_seq(
                    e['src_method'].replace('\n', ' ').replace('\t', ' ').lower(),
                    e['dst_method'].replace('\n', ' ').replace('\t', ' ').lower())
                gnn_input = build_gnn_input(e['src_method'], e['dst_method'])
                e['node_values'] = gnn_input[0]
                e['node_actions'] = gnn_input[1]
                e['edge_prt2ch'] = gnn_input[2]
                e['edge_prev2next'] = gnn_input[3]
                e['edge_align'] = gnn_input[4]
                e['edge_com2sub'] = gnn_input[5]
                sp = spm.SentencePieceProcessor()
                sp.Load("tokenizer/code.model")
                e['src_desc'] = cmt_cleaner.clean(e['src_desc']).replace('\n', ' ').replace('\t', ' ').lower()
                e['dst_desc'] = cmt_cleaner.clean(e['dst_desc']).replace('\n', ' ').replace('\t', ' ').lower()
                e.pop('src_desc_tokens')
                e.pop('dst_desc_tokens')
                e.pop('desc_change_seq')
                # 将e写文件
                with open('data/json/' + file + '_BPE.jsonl', 'a') as fw:
                    b = json.dumps(e)
                    if i != 0:
                        fw.write('\n')
                    fw.write(b)
                    fw.close()


construct_data()
print()
