import string

import jsonlines
from nltk.translate import bleu_score
from rouge import Rouge
from scipy import stats

from eval.SARI import SARIsent

from nlgeval import NLGEval

nlgeval_ = NLGEval()


def eval(src, ref, pred):
    def refine(one_str):
        tran_tab = str.maketrans({key: None for key in string.punctuation})
        new_str = one_str.translate(tran_tab)
        return new_str.lower()

    def cal_acc(r, p):
        if r.lower() == p.lower():
            return 1
        else:
            return 0

    hyp = [pred]
    ref1 = [ref]
    lis = [[r] for r in ref1]
    ans = {}
    try:
        ans = nlgeval_.compute_metrics(hyp_list=hyp, ref_list=lis)
    except:
        pass

    def cal_sari(s, p, r):
        target = [r]
        return SARIsent(s, p, target)

    try:
        ans['SARI'] = cal_sari(src, pred, ref)
    except:
        pass
    try:
        ans['Accuracy'] = cal_acc(ref, pred)
    except:
        pass
    return ans


def eval_TG():
    print('TG')
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

    with open('eval_tools/prediction/tg/output.txt') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i % 4 == 1:
                ref.append(mkstr(line.strip()))
            if i % 4 == 3:
                pred.append(mkstr(line.strip()))

    with open('test_clean_BPE.jsonl') as f:
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
        with open('eval_tools/prediction/tg/' + str(m), 'w') as f:
            for i in m_scores[m]:
                f.write(str(i) + ',')
        to_print = sum(m_scores[m]) / len(m_scores[m])
        print(str(m) + ':\t\t' + str(to_print))
    return m_scores, metrics, pred


def eval_heb():
    print('heb:')
    ref = []
    pred = []
    src = []
    scores = []
    with open('eval_tools/prediction/heb/HebCup_all.json') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if i % 5 == 1:
            src.append(line.split('"')[3])
        if i % 5 == 2:
            ref.append(line.split('"')[3])
        if i % 5 == 3:
            pred.append(line.split('"')[3])

    for i in range(len(ref)):
        scores.append(eval(src[i], ref[i], pred[i]))
    metrics = scores[0].keys()
    m_scores = {}
    for s in scores:
        for m in metrics:
            if m not in m_scores.keys():
                m_scores[m] = [s[m]]
            else:
                m_scores[m].append(s[m])
    for m in metrics:
        with open('eval_tools/prediction/heb/' + str(m), 'w') as f:
            for i in m_scores[m]:
                f.write(str(i) + ',')
        to_print = sum(m_scores[m]) / len(m_scores[m])
        print(str(m) + ':\t\t' + str(to_print))
    return m_scores, metrics, pred


def eval_cup():
    print('cup')
    ref = []
    pred = []
    src = []
    scores = []
    with open('eval_tools/prediction/cup/Baseline_CUP.json') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if i % 5 == 1:
            src.append(line.split('"')[3])
        if i % 5 == 2:
            ref.append(line.split('"')[3])
        if i % 5 == 3:
            pred.append(line.split('"')[3])

    for i in range(len(ref)):
        scores.append(eval(src[i], ref[i], pred[i]))
    metrics = scores[0].keys()
    m_scores = {}
    for s in scores:
        for m in metrics:
            if m not in m_scores.keys():
                m_scores[m] = [s[m]]
            else:
                m_scores[m].append(s[m])
    for m in metrics:
        with open('eval_tools/prediction/cup/' + str(m), 'w') as f:
            for i in m_scores[m]:
                f.write(str(i) + ',')
        to_print = sum(m_scores[m]) / len(m_scores[m])
        print(str(m) + ':\t\t' + str(to_print))
    return m_scores, metrics, pred


def eval_hat():
    print('hat')
    ref = []
    pred = []
    src = []
    scores = []
    with open('eval_tools/prediction/hat/pred.txt') as f:
        lines = f.readlines()
        for line in lines:
            pred.append(line.split('"')[1])
    with open('eval_tools/prediction/hat/ref.txt') as f:
        lines = f.readlines()
        for line in lines:
            ref.append(line.strip())
    with open('eval_tools/prediction/hat/src.txt') as f:
        lines = f.readlines()
        for line in lines:
            src.append(line.strip())

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
                    m_scores[m].append(0)
    for m in metrics:
        with open('eval_tools/prediction/hat/' + str(m), 'w') as f:
            for i in m_scores[m]:
                f.write(str(i) + ',')
        to_print = sum(m_scores[m]) / len(m_scores[m])
        print(str(m) + ':\t\t' + str(to_print))

    return m_scores, metrics, pred


def case_analysis(l, our_p, cup_p, heb_p, hat_p):
    res = []
    with open('test_clean_BPE.jsonl') as f:
        js = list(jsonlines.Reader(f))
        for i in l:
            e = js[i]
            new_e = {'old_code': e['src_method'],
                     'new_code': e['dst_method'],
                     'old_cmt': e['src_desc'],
                     'new_cmt': e['dst_desc'],
                     'TG': our_p[i],
                     'cup': cup_p[i],
                     'heb': heb_p[i],
                     'hat': hat_p[i]}
            res.append(new_e)
    return res


def eval_all():
    T63, metrics, our_pred = eval_TG()
    cup, _, cup_pred = eval_cup()
    heb, _, heb_pred = eval_heb()
    hat, _, hat_pred = eval_hat()
    cup_acc_list = cup['Accuracy']
    heb_acc_list = heb['Accuracy']
    hat_acc_list = hat['Accuracy']
    our_acc_list = T63['Accuracy']
    for m in metrics:
        our = T63[m]
        cups = cup[m]
        hebs = heb[m]
        hats = hat[m]
        our_nciu = []
        cup_nciu = []
        heb_nciu = []
        hat_nciu = []
        for i, a in enumerate(heb_acc_list):
            if a == 0:
                our_nciu.append(our[i])
                cup_nciu.append(cups[i])
                heb_nciu.append(hebs[i])
                hat_nciu.append(hats[i])
        try:
            _, p_value = stats.wilcoxon(our, cups)
            print('cup nciu:', sum(cup_nciu) / len(cup_nciu))
            print(str(m) + ': cup -p:', p_value)
        except:
            print(str(m) + ': cup -p: False')
        try:
            _, p_value = stats.wilcoxon(our, hebs)
            print('heb nciu:', sum(heb_nciu) / len(heb_nciu))
            print(str(m) + ': heb -p:', p_value)
        except:
            print(str(m) + ': heb -p: False')
        try:
            _, p_value = stats.wilcoxon(our, hats)
            print('hat nciu:', sum(hat_nciu) / len(hat_nciu))
            print(str(m) + ': hat -p:', p_value)
        except:
            print(str(m) + ': hat -p: False')
        print(str(m) + ': our nciu:', sum(our_nciu) / len(our_nciu))

    only_our_true = []
    our_fail = []
    for i in range(len(cup_acc_list)):
        if our_acc_list[i] == 0:
            our_fail.append(i)
        if cup_acc_list[i] == 0 and heb_acc_list[i] == 0 and hat_acc_list[i] == 0 and our_acc_list[i] == 1:
            only_our_true.append(i)
    return only_our_true, our_fail, our_pred, cup_pred, heb_pred, hat_pred


def run():
    only_our_true, our_fail, our_pred, cup_pred, heb_pred, hat_pred = eval_all()
    only_our_true_es = case_analysis(
        only_our_true, our_pred, cup_pred, heb_pred, hat_pred)
    our_fail_es = case_analysis(
        our_fail, our_pred, cup_pred, heb_pred, hat_pred)
    with open('eval_tools/only_our_true_es', 'w') as f:
        for i in only_our_true_es:
            for key in i.keys():
                f.write(str(key) + ':\n')
                f.write(str(i[key]) + '\n')
            f.write('------------------------------------\n')
    with open('eval_tools/our_fail_es', 'w') as f:
        for i in our_fail_es:
            for key in i.keys():
                f.write(str(key) + ':\n')
                f.write(str(i[key]) + '\n')
            f.write('------------------------------------\n')
