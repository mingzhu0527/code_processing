import os
import torch
device = torch.device("cuda")

import sys
sys.path.append('./huggingface_models/')
sys.path.append('./utils/')
from sample_utils import *
from inference_utils import *
from codenet_process_utils import *
from self_training_utils import *

def get_preds_lang_dict_codenet(lang_pairs, model_type, device, programs_dict, 
                        sample_size=5, temperature=0.5,
                        data_name="xlcost", tag='test', exp_suffix="_translation_exec_function/"):
    preds_lang_dict = {}
    for lang1, lang2 in lang_pairs:
        print(lang1, lang2)
        src_codes = programs_dict[lang1]
        tgt_codes = []
        is_eval = False
        eval_examples, eval_features, eval_dataloader, model, tokenizer, args, decoder_sid = inference_prepro(
             lang1, lang2, model_type, device, src_codes, tgt_codes, None, tag, exp_suffix)
        preds, eval_result = generation_multiple(eval_examples, eval_dataloader, 
                                                     model, tokenizer, args, device, 
                                                     decoder_sid, is_eval, sample_size, temperature)
        preds_lang_dict[(lang1, lang2)] = preds
        preds_lang_dict_path = cached_path + model_type + "_" + tag + "_" + lang1 + "_" \
                    + data_name + "_1preds_lang_dict.pkl"
        with open(preds_lang_dict_path, 'wb') as outfile:
            pickle.dump(preds_lang_dict, outfile)
    return preds_lang_dict

data_name = "codenet"
is_eval = False
sample_size = 10
temperature = 0.5
tag = "full"
model_type = "codet5"
exp_suffix = "_translation_exec_function/" 


is_plbart = model_type=="plbart"
merged_filtered_dict = get_prepro_filtered_dict(None, is_plbart)

programs_dict = {}
for lang in new_langs:
    functions = []
    for dic in merged_filtered_dict[lang]:
        functions.append(dic['function_notok'])
    programs_dict[lang] = functions

new_langs = ['C#', 'Java', 'Python', 'C++']

lang_pairs = []
for lang1 in new_langs:
    for lang2 in new_langs:
        if lang2 != lang1:
            lang_pairs.append((lang1, lang2))
small_programs_dict = {x:["haha"] for x in new_langs}

preds_lang_dict = get_preds_lang_dict_codenet(lang_pairs, model_type, device, programs_dict, 
                                      sample_size, temperature, data_name, tag, exp_suffix)