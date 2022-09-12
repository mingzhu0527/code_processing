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

model_type = "codet5"
is_plbart = model_type=="plbart"

merged_filtered_dict_path = cached_path + 'archived_files/codenet_merged_filtered_dict_notok.json'
with open(merged_filtered_dict_path, 'rb') as infile:
     merged_filtered_dict = json.load(infile)
# merged_filtered_dict = get_prepro_filtered_dict(None, is_plbart)
programs_dict = get_codenet_programs_dict(merged_filtered_dict)

import_str_dict = {}
for lang in new_langs:
    all_imports, import_str = get_common_imports(lang, merged_filtered_dict)
    import_str_dict[lang] = import_str
import_str_dict["Java"] = java_imports_str
import_str_dict["C#"] = csharp_imports_str

plbart_sample_path = cached_path + 'codet5/last_batch/codet5_codenet_preds_lang_dict.pkl'
with open(plbart_sample_path, 'rb') as infile:
     preds_lang_dict_plbart = pickle.load(infile)

new_langs = ['C++', 'Java', 'C#', 'Python']

lang_pairs = []
for lang1 in new_langs:
    for lang2 in new_langs:
        if lang2 != lang1:
            lang_pairs.append((lang1, lang2))
call_dict = {}
for lang1, lang2 in lang_pairs:
    print(lang1, lang2)
    preds = preds_lang_dict_plbart[(lang1, lang2)]
    # remove duplicated preds
    new_preds = get_dedup_preds(preds)
    # filter by type-matching (Todo)
    functions, function_id_dict = prep_exec_hypo_codenet(new_preds, lang1, lang2, 
                                                                 merged_filtered_dict, model_type)
    
    # filter by compilation
    call_list = get_hypo_call_list(functions, lang2, import_str_dict)
    call_dict[(lang1, lang2)] = [new_preds, functions, function_id_dict, call_list]
    with open(cached_path + "codet5_old_no_C_codenet_lang_pair_call_dict.pkl", 'wb') as outfile:
        pickle.dump(call_dict, outfile)