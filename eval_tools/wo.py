import jsonlines
from utils import eval

def eval_wo():
    print('w/o')
    ref = []
    pred = []
    src = []
    scores = []

    def mkstr(s):
        if len(s) > 1:
            if s[-1] == '.' and s[-2] != ' ':
                return s[:-1] + ' .'
            else:
                return s
        else:
            return s

    with open('/Users/chenyn/研/论文撰写/Updater/实验/tools/eval_tools/prediction/wo/output.txt') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i % 4 == 1:
                ref.append(mkstr(line.strip()))
            if i % 4 == 3:
                pred.append(mkstr(line.strip()))

    with open('/Users/chenyn/研/论文撰写/Updater/实验/Transformer实现/data/json/test_clean_BPE.jsonl') as f:
        js = list(jsonlines.Reader(f))
        for e in js:
            src.append(mkstr(e['src_desc'].strip()))

    for i in range(len(ref)):
        scores.append(eval(src[i], ref[i], pred[i]))
    metrics = scores[0].keys()
    m_scores = {}
    for s in scores:
        for m in metrics:
            if m not in m_scores.keys():
                m_scores[m] = [s[m]]
            else:
                if m in s.keys():
                    m_scores[m].append(s[m])
                else:
                    m_scores[m].append(1)
    for m in metrics:
        with open('/Users/chenyn/研/论文撰写/Updater/实验/tools/eval_tools/prediction/wo/' + str(m), 'w') as f:
            for i in m_scores[m]:
                f.write(str(i) + ',')
        to_print = sum(m_scores[m]) / len(m_scores[m])
        print(str(m) + ':\t\t' + str(to_print))
    return m_scores, metrics, pred

eval_wo()