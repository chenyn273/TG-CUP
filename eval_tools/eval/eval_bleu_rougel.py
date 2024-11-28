import sacrebleu
from nltk.translate import bleu_score
from rouge import Rouge


def cal_bleu(ref, pred):
    reference = [ref]
    candidate = pred
    return bleu_score.sentence_bleu(reference, candidate)


def cal_rouge(ref, pred):
    rouge = Rouge()
    return rouge.get_scores(pred, ref, avg=True)['rouge-l']['f']


def main():
    pred = []
    ref = []
    with open('/Users/chenyn/研/论文撰写/Updater/实验/tools/eval_tools/prediction/heb/pred_1.txt') as f:
        lines = f.readlines()
        for line in lines:
            pred.append(line.strip())

    with open('/Users/chenyn/研/论文撰写/Updater/实验/tools/eval_tools/prediction/heb/ref.txt') as f:
        lines = f.readlines()
        for line in lines:
            ref.append(line.strip())
    bleu_list = []
    rouge_list = []
    for i, p in enumerate(pred):
        try:
            bleu_list.append(cal_bleu(ref[i], p))
        except:
            pass
        try:
            rouge_list.append(cal_rouge(ref[i], p))
        except:
            pass
    print(sum(bleu_list) / len(bleu_list))
    print(sum(rouge_list) / len(rouge_list))


main()
