from code_prepro.bpe_modes.fast_bpe_mode import FastBPEMode
import os


def get_bpe(is_roberta=False):
    bpe_model = FastBPEMode(codes=os.path.abspath(Fast_codes), vocab_path=None)
#     dico = Dictionary.read_vocab(Fast_vocab)
    if is_roberta:
        bpe_model = RobertaBPEMode()
#         dico = Dictionary.read_vocab(Roberta_BPE_path)
    return bpe_model

Fast_BPE_path = "./code_prepro/bpe/cpp-java-python/"
Fast_codes = Fast_BPE_path + 'codes'
Fast_vocab = Fast_BPE_path + 'vocab'
Roberta_BPE_path = "./code_prepro/bpe/roberta-base-vocab"
