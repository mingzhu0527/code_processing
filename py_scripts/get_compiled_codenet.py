import sys
sys.path.append('./huggingface_models/')
sys.path.append('./utils/')
from sample_utils import *
from inference_utils import *
from codenet_process_utils import *
from self_training_utils import *

num_batch = 41
todo_idx = [i for i in range(25, num_batch)] + [6]
# new_langs = ['C']
for i in todo_idx:
    print(i)
    codedict_path = cached_path + 'codenet/codenet_codedict_' + str(i) + '.json'
    if os.path.exists(codedict_path):
        with open(codedict_path) as infile:
            code_lang_dict = json.load(infile)
    func_id_dict, program_dict, imports_dict = get_nonempty_functions(code_lang_dict, new_langs)
    call_dict = get_codenet_call_dict(program_dict, imports_dict, new_langs)
#     call_dict_list.append(call_dict)
    filtered_dict = get_compiled_functions(call_dict, func_id_dict, imports_dict, program_dict, 
                                       code_lang_dict)
    call_dict_path = cached_path + 'codenet/codenet_call_dict_' + str(i) + '.json'
    with open(call_dict_path, 'w') as outfile:
        json.dump(call_dict, outfile)
#     filtered_dict_list.append(filtered_dict)
    filtered_dict_path = cached_path + 'codenet/codenet_filtered_dict_' + str(i) + '.json'
    with open(filtered_dict_path, 'w') as outfile:
        json.dump(filtered_dict, outfile)