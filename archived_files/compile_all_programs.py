import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device = torch.device("cuda")
import sys
sys.path.append('./huggingface_models/')
from run import *
from compilation_utils import *
from tokenization_utils import *
from bleu import *

# def compile_by_split():
    

# Test compilation of original data
# split_dict = load_split_dict()
# test_list, val_list = get_eval_list(split_dict)
program_json_dict, program_id_lang_dic = read_program_tok_file()

# test_set = set(test_list)
# val_set = set(val_list)
# eval_set = test_set | val_set
error_path = './'
cr_dict = {}
for lang, lang_dic in program_id_lang_dic.items():
    print(lang)
    if lang in ['Java', 'C++', 'C', 'Python', "Javascript"]:
        continue
    keys = list(lang_dic.keys())
#     train_set = set(keys) - eval_set
#     train_list = list(train_set)
#     keys_list = [train_list, val_list, test_list]
#     cr_lang_dict = {tag:[] for tag in tags}
#     for tag, keys in zip(tags, keys_list):
#         print(tag)
    compile_programs, compilation_rate, compile_programs_detoked, compilation_rate_detoked \
        = get_compilation_by_split(program_id_lang_dic, lang, keys)
#     cr_lang_dict[tag] = [compilation_rate, compilation_rate_detoked]
    cr_dict[lang] = [compilation_rate, compilation_rate_detoked]
    print(cr_dict)
    error_fn = lang + ".error_report.jsonl"
    error_fn_detoked = lang + "-detoked.error_report.jsonl"
    save_compilation_report(error_path, error_fn, compile_programs)
    save_compilation_report(error_path, error_fn_detoked, compile_programs_detoked)
        
with open(error_path + "cr_dict.json", 'w') as outfile:
    json.dump(cr_dict, outfile)
    

