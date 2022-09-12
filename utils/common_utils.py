from code_prepro.lang_processors import *
from code_prepro.bpe_modes import *


import json
from tqdm import tqdm
import shutil
from shutil import copyfile
import re
import os
import random
from sklearn.model_selection import train_test_split
import random
import jsonlines
import pickle
import numpy as np
import unicodedata
import chardet

def get_tokenizer(lang):
    processor = LangProcessor.processors[lang](root_folder=so_path)
    tokenizer = processor.tokenize_code
    return tokenizer

def get_detokenizer(lang):
    processor = LangProcessor.processors[lang](root_folder=so_path)
    tokenizer = processor.detokenize_code
    return tokenizer

def get_bpe(is_roberta=False):
    bpe_model = FastBPEMode(codes=os.path.abspath(Fast_codes), vocab_path=None)
#     dico = Dictionary.read_vocab(Fast_vocab)
    if is_roberta:
        bpe_model = RobertaBPEMode()
#         dico = Dictionary.read_vocab(Roberta_BPE_path)
    return bpe_model

home_path = "/home/mingzhu/CodeModel/"
evaluator_path = home_path + "CodeXGLUE/Code-Code/code-to-code-trans/evaluator/"
data_path = home_path + "g4g/XLCoST_data/"
cwd_path = home_path + "CodeGen_cwd/"

code_prepro_path = cwd_path + "code_prepro/"
so_path = code_prepro_path + "lang_processors/"
Fast_BPE_path = home_path + "CodeGen/data/bpe/cpp-java-python/"
# Fast_BPE_path = code_prepro_path + "data/bpe/cpp-java-python/"
Roberta_BPE_path = home_path + "CodeGen/data/bpe/roberta-base-vocab"
# Roberta_BPE_path = code_prepro_path + "data/bpe/roberta-base-vocab"
Fast_codes = Fast_BPE_path + 'codes'
Fast_vocab = Fast_BPE_path + 'vocab'
# so_path = home_path + "CodeGen/codegen_sources/preprocessing/lang_processors"

map_data_path = data_path + "map_data/"
split_dict_path = map_data_path + "split_dict.json"

snippet_data = data_path + "pair_data_tok_1/"
program_data = data_path + "pair_data_tok_full/"
mono_snippet_data = data_path + "pair_data_tok_1_comment/"
mono_program_data = data_path + "pair_data_tok_full_desc_comment/"
function_data = data_path + "pair_data_tok_function/"
function_map_data_path = data_path + "functions_map_data/"
code_dict_path = data_path + "functions_code_dict/"

cached_path = cwd_path + 'cached_files/'
pids_dict_path = cached_path + "pids_dict.json"
programs_dict_path = cached_path + "programs_dict.json"
programs_toked_dict_path = cached_path + "programs_toked_dict.json"
programs_detoked_dict_path = cached_path + "programs_detoked_dict.json"

functions_pids_dict_path = cached_path + "functions_pids_dict.json"
functions_dict_path = cached_path + "functions_dict.json"
functions_toked_dict_path = cached_path + "functions_toked_dict.json"
functions_detoked_dict_path = cached_path + "functions_detoked_dict.json"
sys_calls_path = cached_path+"sys_calls_dict.json"

function_dict = {'Java':'method_declaration', 'Python':'function_definition', 'C++':'function_definition',
                'C#':'method_declaration', 'Javascript':'function_declaration', 'PHP':'function_definition', 'C':'function_definition'}
function_call_dict = {'Java':'method_invocation', 'Python':'call', 'C++':'call_expression',
                'C#':'invocation_expression', 'Javascript':'call_expression', 'PHP':'function_call_expression', 'C':'call_expression'}
target_function_place_holder = "\n<target_function_place_holder>\n"
escape_list = ['printArray', 'newNode']

# data_path = "/home/mingzhu/CodeModel/g4g/XLCoST_functions/"
# data_path = "/home/mingzhu/CodeModel/g4g/archived_data/XLCoST_functions/"
code_viewer_path = home_path + "CodeGen/code_viewer/"
# so_path = "/home/mingzhu/CodeModel/CodeGen/codegen_sources/preprocessing/lang_processors/"
# dump_path = home_path + "CodeGen/dumppath1/"

file_extensions = {"Java": ".java", "C++": ".cpp", "C": ".c", "Python": ".py","Javascript": ".js",
                   "PHP":".php", "C#":".cs"}
lang_lower = {"Java": "java", "C++": "cpp", "C": "c", "Python": "python","Javascript": "javascript",
                   "PHP":"php", "C#":"csharp"}
lang_map = {"Java": "java", "C++": "cpp", "C": "c", "Python": "python","Javascript": "javascript",
                   "PHP":"php", "C#":"c_sharp"}
lang_upper = {"java": "Java", "cpp": "C++", "c": "C", "python": "Python","javascript": "Javascript",
                   "php":"PHP", "csharp":"C#"}
tags = ['train', 'val', 'test']

lang_py = 'python'
lang_java = 'java'
lang_cs = 'csharp'
lang_cpp = 'cpp'
lang_c = 'c'
lang_php = 'php'
lang_js = 'javascript'
langs = ["C++", "Java", "Python", "C#", "Javascript", "PHP", "C"]
new_langs = ["C++", "Java", "Python", "C#", "C"]



bpe_model = get_bpe()
py_tokenizer = get_tokenizer(lang_py)
cs_tokenizer = get_tokenizer(lang_cs)
java_tokenizer = get_tokenizer(lang_java)
cpp_tokenizer = get_tokenizer(lang_cpp)
# js_tokenizer = get_tokenizer(lang_js)
js_tokenizer = get_tokenizer(lang_java)
c_tokenizer = get_tokenizer(lang_c)
# php_tokenizer = get_tokenizer(lang_php)
php_tokenizer = c_tokenizer

py_detokenizer = get_detokenizer(lang_py)
cs_detokenizer = get_detokenizer(lang_cs)
java_detokenizer = get_detokenizer(lang_java)
cpp_detokenizer = get_detokenizer(lang_cpp)
# js_detokenizer = get_detokenizer(lang_js)
js_detokenizer = get_detokenizer(lang_java)
c_detokenizer = get_detokenizer(lang_c)
# php_tokenizer = get_detokenizer(lang_php)
php_detokenizer = c_detokenizer

file_tokenizers = {"Java": java_tokenizer, "C++": cpp_tokenizer, "C": c_tokenizer, "Python": py_tokenizer,
                   "Javascript": js_tokenizer, "PHP": php_tokenizer, "C#": cs_tokenizer}
file_detokenizers = {"Java": java_detokenizer, "C++": cpp_detokenizer, "C": c_detokenizer, "Python": py_detokenizer,
                   "Javascript": js_detokenizer, "PHP": php_detokenizer, "C#": cs_detokenizer}
        
baseline_model_dict = {'plbart':"uclanlp/plbart-python-en_XX", 'codet5':"Salesforce/codet5-base", 
                      'codebert':"microsoft/codebert-base", 'graphcodebert':"microsoft/graphcodebert-base", 
                      'roberta':'roberta-base'}