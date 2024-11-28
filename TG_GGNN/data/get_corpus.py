import jsonlines

import json
from cup_utils.comment import CommentCleaner

if __name__ == "__main__":

    files = ['test_clean', 'train_clean', 'valid_clean']
    corpus_path = 'corpus.code'
    corpus_lines = []
    cmt_cleaner = CommentCleaner(True)

    for file in files:

        with open('./json/' + file + '.jsonl', 'r+', encoding='utf8') as f:
            js = list(jsonlines.Reader(f))
            for e in js:
                if str(e).lower().__contains__('ggfs') or str(e).lower().__contains__('igfs'):
                    continue
                corpus_lines.append(e['src_method'].replace(
                    '\n', ' ').replace('\t', ' ').lower() + '\n')
                corpus_lines.append(e['dst_method'].replace(
                    '\n', ' ').replace('\t', ' ').lower() + '\n')
                corpus_lines.append(
                    cmt_cleaner.clean(e['src_desc']).replace('\n', ' ').replace('\t', ' ').lower() + '\n')
                corpus_lines.append(
                    cmt_cleaner.clean(e['dst_desc']).replace('\n', ' ').replace('\t', ' ').lower() + '\n')

    with open(corpus_path, 'w') as f:
        f.writelines(corpus_lines)

    print("lines of corpus: ", len(corpus_lines))
