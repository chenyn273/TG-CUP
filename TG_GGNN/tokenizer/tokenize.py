import sentencepiece as spm


def train(input_file, vocab_size, model_name, model_type, character_coverage):
    """
    search on https://github.com/google/sentencepiece/blob/master/doc/options.md to learn more about the parameters
    :param input_file: one-sentence-per-line raw corpus file. No need to run tokenizer, normalizer or preprocessor.
                       By default, SentencePiece normalizes the input with Unicode NFKC.
                       You can pass a comma-separated list of files.
    :param vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
    :param model_name: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
    :param model_type: model type. Choose from unigram (default), bpe, char, or word.
                       The input sentence must be pretokenized when using word type.
    :param character_coverage: amount of characters covered by the model, good defaults are: 0.9995 for languages with
                               rich character set like Japanse or Chinese and 1.0 for other languages with
                               small character set.
    """
    input_argument = '--user_defined_symbols="<Node>","NodeInsert","NodeDelete","NodeNone","NodeMove","NodeUpdate","<old_cmt_st>","<old_cmt_ed>","<code_edit_st>","<code_edit_ed>","<before>","<after>","<c_insert>","<c_delete>","<c_replace>","<c_keep>","<n_insert>","<n_delete>","<n_replace>","<n_keep>" ' \
                     '--input=%s --model_prefix=%s --vocab_size=%s --model_type=%s --character_coverage=%s ' \
                     '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 '
    cmd = input_argument % (input_file, model_name, vocab_size, model_type, character_coverage)
    print(cmd)
    spm.SentencePieceTrainer.Train(cmd)


def run():
    # en_input = '../data/corpus.en'
    # en_vocab_size = 32000
    # en_model_name = 'eng'
    # en_model_type = 'bpe'
    # en_character_coverage = 1
    # train(en_input, en_vocab_size, en_model_name, en_model_type, en_character_coverage)
    #
    # ch_input = '../data/corpus.ch'
    # ch_vocab_size = 32000
    # ch_model_name = 'chn'
    # ch_model_type = 'bpe'
    # ch_character_coverage = 0.9995
    # train(ch_input, ch_vocab_size, ch_model_name, ch_model_type, ch_character_coverage)

    code_input = '../data/corpus.code'
    code_vocab_size = 32000
    code_model_name = 'code'
    code_model_type = 'bpe'
    code_character_coverage = 0.9995
    train(code_input, code_vocab_size, code_model_name, code_model_type, code_character_coverage)


def test():
    # sp = spm.SentencePieceProcessor()
    # text = "美国总统特朗普今日抵达夏威夷。"
    #
    # sp.Load("./chn.model")
    # print(sp.EncodeAsPieces(text))
    # print(sp.EncodeAsIds(text))
    # a = [12907, 277, 7419, 7318, 18384, 28724]
    # print(sp.decode_ids(a))
    sp = spm.SentencePieceProcessor()
    text = "Save cluster basic configuration."
    sp.Load("./code.model")
    print(sp.EncodeAsPieces(text))
    # print(sp.EncodeAsIds(text))
    # a = [13046, 1462, 3345, 873, 31934]
    # print(sp.decode_ids(a))


if __name__ == "__main__":
    run()
    # test()
