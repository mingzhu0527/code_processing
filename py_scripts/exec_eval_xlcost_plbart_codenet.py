import os
import torch
device = torch.device("cuda")

import sys
sys.path.append('./huggingface_models/')
sys.path.append('./utils/')
from self_training_utils import *

new_langs = ["C++", "Java", "Python", "C#", "C"]

lang_pairs = []
for lang1 in new_langs:
    for lang2 in new_langs:
        if lang2 != lang1:
            lang_pairs.append((lang1, lang2))
            
data_name = 'xlcost'
tag = "test"
function_data_path = home_path + "g4g/XLCoST_data/pair_data_notok_exec_function/"
data_code_dict_path = code_dict_path

do_sample, temperature = True, 0.5
sample_size = 5
model_type = "plbart"
exp_suffix = "_translation_transfer_codenet_exec_function/" # _translation_transfer_codenet_exec_function 
# _translation_exec_codenet_function
# exp_suffix = "_translation_exec_codenet_small_function/"
# lang_pairs = [("Java", "C#"), ("C#", "Java"), ("Java", "Python"), ("Python", "Java"),
#              ("Java", "C++"), ("C++", "Java"), ("Python", "C++"), ("C++", "Python"),
#              ("Python", "C#"), ("C#", "Python"), ("C#", "C++"), ("C++", "C#")]
lang_pairs = [('Java', "C"), ('Python', "C"), ('C#', "C"), ('C++', "C"),
              ('C', 'Java'), ('C', 'Python'), ('C', 'C#'), ('C', 'C++')]
label = "_exec_codenet_transfer"

# model_type = sys.argv[1]

code_lang_dict, code_id_lang_dic = read_toked_code_dict(data_code_dict_path)
call_dict_gold = get_call_dict_gold(lang_pairs, function_data_path, code_id_lang_dic, data_name, tag)

preds_lang_dict = get_preds_lang_dict_new(lang_pairs, model_type, device, None, None, function_data_path, 
                                      sample_size, temperature, data_name, tag, exp_suffix, label, do_sample)
call_dict_hypo = get_call_dict_hypo(lang_pairs, preds_lang_dict, call_dict_gold, code_id_lang_dic, 
                                    model_type, data_name, tag, label)
hypo_filtering_dict = get_hypo_filtering_dict(lang_pairs, call_dict_gold, call_dict_hypo)