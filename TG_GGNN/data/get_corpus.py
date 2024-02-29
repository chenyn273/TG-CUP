import jsonlines

import json
from cup_utils.comment import CommentCleaner

if __name__ == "__main__":
    # files = ['train', 'dev', 'test']
    # ch_path = 'corpus.ch'
    # en_path = 'corpus.en'
    # ch_lines = []
    # en_lines = []
    files = ['test_clean', 'train_clean', 'valid_clean']
    corpus_path = 'corpus.code'
    corpus_lines = []
    cmt_cleaner = CommentCleaner(True)

    for file in files:
        # corpus = json.load(open('./json/' + file + '.jsonl', 'r'))
        # for item in corpus:
        #     ch_lines.append(item[1] + '\n')
        #     en_lines.append(item[0] + '\n')
        with open('./json/' + file + '.jsonl', 'r+', encoding='utf8') as f:
            js = list(jsonlines.Reader(f))
            for e in js:
                if str(e).lower().__contains__('ggfs') or str(e).lower().__contains__('igfs'):
                    continue
                corpus_lines.append(e['src_method'].replace('\n', ' ').replace('\t', ' ').lower() + '\n')
                corpus_lines.append(e['dst_method'].replace('\n', ' ').replace('\t', ' ').lower() + '\n')
                corpus_lines.append(
                    cmt_cleaner.clean(e['src_desc']).replace('\n', ' ').replace('\t', ' ').lower() + '\n')
                corpus_lines.append(
                    cmt_cleaner.clean(e['dst_desc']).replace('\n', ' ').replace('\t', ' ').lower() + '\n')

    # with open(ch_path, "w") as fch:
    #     fch.writelines(ch_lines)

    # with open(en_path, "w") as fen:
    #     fen.writelines(en_lines)
    with open(corpus_path, 'w') as f:
        f.writelines(corpus_lines)

    # lines of Chinese: 252777
    # print("lines of Chinese: ", len(ch_lines))
    # # lines of English: 252777
    # print("lines of English: ", len(en_lines))
    # print("-------- Get Corpus ! --------")
    print("lines of corpus: ", len(corpus_lines))
