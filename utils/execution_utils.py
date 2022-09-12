import subprocess
import shlex
import time
import os
import stat
import random
import re
from p_tqdm import p_map, p_umap, p_imap, p_uimap
from extract_function_utils import *

def get_call_dict_hypo_codex(lang_pairs, preds_lang_dict, pids_dict, 
                       call_dict_gold, code_id_lang_dic, 
                       model_type, data_name="xlcost", tag="test", label=""):
    call_dict_hypo_path = cached_path + \
        model_type + "_" + tag + "_" + data_name + "_call_dict_hypo" + label + ".pkl"
    if os.path.exists(call_dict_hypo_path):
        with open(call_dict_hypo_path, 'rb') as infile:
            call_dict_hypo = pickle.load(infile)
            return call_dict_hypo
    call_dict_hypo = {}
    for lang1, lang2 in lang_pairs:
        print(lang1, lang2)
        preds = preds_lang_dict[(lang1, lang2)]
#         pids = call_dict_gold[(lang1, lang2)][1]
        pids = pids_dict[(lang1, lang2)]
        programs, program_id_dict, program_dict = prep_exec_hypo(preds, pids, code_id_lang_dic, lang2, model_type)
        lang_results = p_map(file_executors[lang2], programs)
        result_type_dict = show_result_summary({lang2:lang_results})
        result_dicts = result_mapping(lang_results, program_id_dict, pids, lang2)
        result_id_dict, result_key_dict, error_type_dict = result_dicts
        call_dict_hypo[(lang1, lang2)] = [pids, programs, program_id_dict, program_dict, 
                                          (result_id_dict, result_key_dict, error_type_dict)]
    with open(call_dict_hypo_path, 'wb') as outfile:
        pickle.dump(call_dict_hypo, outfile)
    return call_dict_hypo


def get_call_dict_hypo(lang_pairs, preds_lang_dict, call_dict_gold, code_id_lang_dic, 
                       model_type, data_name="xlcost", tag="test", label=""):
    call_dict_hypo_path = cached_path + model_type + "_" + tag + "_" + data_name + "_call_dict_hypo" + label + ".pkl"
    if os.path.exists(call_dict_hypo_path):
        with open(call_dict_hypo_path, 'rb') as infile:
            call_dict_hypo = pickle.load(infile)
            return call_dict_hypo
    call_dict_hypo = {}
    for lang1, lang2 in lang_pairs:
        print(lang1, lang2)
        preds = preds_lang_dict[(lang1, lang2)]
        pids = call_dict_gold[(lang1, lang2)][1]
        programs, program_id_dict, program_dict = prep_exec_hypo(preds, pids, code_id_lang_dic, lang2, model_type)
        lang_results = p_map(file_executors[lang2], programs)
        result_type_dict = show_result_summary({lang2:lang_results})
        result_dicts = result_mapping(lang_results, program_id_dict, pids, lang2)
        result_id_dict, result_key_dict, error_type_dict = result_dicts
        call_dict_hypo[(lang1, lang2)] = [pids, programs, program_id_dict, program_dict, 
                                          (result_id_dict, result_key_dict, error_type_dict)]
    with open(call_dict_hypo_path, 'wb') as outfile:
        pickle.dump(call_dict_hypo, outfile)
    return call_dict_hypo

def get_call_dict_gold(lang_pairs, function_data_path, code_id_lang_dic, data_name="xlcost", tag='test'):
    call_dict_gold_path = cached_path + tag + "_" + data_name +  "_call_dict_gold.pkl"
    if os.path.exists(call_dict_gold_path):
        with open(call_dict_gold_path, 'rb') as infile:
            call_dict_gold = pickle.load(infile)
            return call_dict_gold
    call_dict_gold = {}
    for lang1, lang2 in lang_pairs:
        print(lang1, lang2)
        map_dict, reverse_map_dict = get_target_map_dict(lang1, lang2, function_data_path, tag)
        pids = [map_dict[x] for x in range(len(map_dict))]
        programs_gold, lang_results_gold, result_dicts = exec_gold(map_dict, lang2, code_id_lang_dic)
        result_id_dict_gold, result_key_dict_gold, error_type_dict_gold = result_dicts
        call_dict_gold[(lang1, lang2)] =  [(map_dict, reverse_map_dict), pids, programs_gold, 
                                           (result_id_dict_gold, result_key_dict_gold, error_type_dict_gold)]
    return call_dict_gold

def show_result_summary(result_dict):
    dics = exec_result_analysis(result_dict)
    result_dic_names = ['error', 'timeout', 'empty', 'other', 'good']
    for lang in result_dict.keys():
        for name, dic in zip(result_dic_names, dics):
            if len(result_dict[lang]) > 0:
                print(lang, name, len(dic[lang]), len(dic[lang])*1.0/len(result_dict[lang]))
    result_type_dict = {x:dic for x, dic in zip(result_dic_names, dics)}
    return result_type_dict

def get_error_types(results, lang):
    error_type_dict = {}
    result_keys = []
    processed_results = []
    for i, error_msg in enumerate(results):
        if lang == "C++":
            error_msg = error_msg.encode("ascii", "ignore").decode()
        processed_results.append(error_msg)
        result_key = result_class(error_msg)
        result_keys.append(result_key)
        if result_key == 'error':
            error_type, error_line = error_class(error_msg, lang)
            if error_type in error_type_dict:
                error_type_dict[error_type] += 1
            else:
                error_type_dict[error_type] = 1
    sorted_error_type_dict = dict(sorted(error_type_dict.items(), key=lambda item: item[1])[-10:])
    return processed_results, result_keys, sorted_error_type_dict

def result_mapping(results, program_id_dict, pids, lang="Python"):
    result_id_dict = {}
    result_key_dict = {}
    processed_results, result_keys, error_type_dict = get_error_types(results, lang)
    for pid in pids:
        result_id_dict[pid] = []
        result_key_dict[pid] = []
        for idx in program_id_dict[pid]:
            result_id_dict[pid].append(processed_results[idx])
            result_key_dict[pid].append(result_keys[idx])
    return result_id_dict, result_key_dict, error_type_dict        

def get_python_imports():
    imports_list = ['sys', 'time', 'collections', 'itertools', 
                    'math', 'random', 'operator', 'fractions', 'heapq']
    imports_irregular_list = ['from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2', 
                         "from collections import defaultdict",
                        'from itertools import accumulate, product, permutations, combinations',
                        'from collections import Counter, OrderedDict, deque, defaultdict, ChainMap',
                        'from functools import lru_cache',
                         'from typing import List, Tuple',
                         'import numpy as np',
                         'from heapq import *']
    import_str = ""
    for imp in imports_list:
        import_str += "import " + imp + "\n"
    import_str += '\n'.join(imports_irregular_list) + '\n'
    return import_str

# this one doesn't have testcase. Need another one for testcase
def get_hypo_callcode(samples, pid, lang, function_id_lang_dic, is_plbart=False):
    code_dic = function_id_lang_dic[lang][pid]
    target_func= code_dic['target_call']
    program_l = []
    for sample in samples:
        sample = notok_detok(sample, lang, is_plbart)
        hypo_target_func = get_hypo_target_fn(sample, target_func, lang)
        target_hypo = sample.replace(hypo_target_func, target_func)
        program = "".join(code_dic['program_pieces']).replace(target_function_place_holder, 
                                                                           target_hypo)
        program_l.append(program)
    return program_l

# to get callcode for any lang:
# just get target function name and replace with the groundtruth one.
# other language also doesn't need the detok
# However, to eval on testcases, need the callcode function.
def get_hypo_callcode_python(samples, pid, lang, function_id_lang_dic):
    import_str = get_python_imports()
    code_dic = function_id_lang_dic[lang][pid]
    target_func= code_dic['target_call']
    program_l = []
    for sample in samples:
        sample_detok = py_detokenizer(sample)
        hypo_target_func = get_hypo_target_fn(sample_detok, target_func, lang)
        target_hypo = sample_detok.replace(hypo_target_func, target_func)
        program = import_str + "".join(code_dic['program_pieces']).replace(target_function_place_holder, 
                                                                           target_hypo)
        program_l.append(program)
    return program_l

def get_func_names_python(code_string):
    lines = code_string.split('\n')
    fns = []
    for line in lines:
        line_parts = line.strip().split(' ', 1)
        if len(line_parts) >1 and line_parts[0] == 'def':
            try:
                fn = line_parts[1].split('(', 1)[0]
                fns.append(fn)
            except:
                continue
    return fns

def get_hypo_function_info(code_string, lang):
    if lang == "Python":
        fns = get_func_names_python(code_string)
        return fns
    else:
        code1 =  get_code_for_parsing(lang, code_string)
        fns, params, return_types = extract_function_info(code1, lang)
        return (fns, params, return_types)
        
def get_hypo_target_info(code_string, target_func, lang):
    function_info = get_hypo_function_info(code_string, lang)
#     code1 =  get_code_for_parsing(lang, code_string)
    hypo_target_func = ""
    target_param = None
    target_return_type = None
    if lang == "Python":
        fns = function_info
#         fns = get_func_names_python(code_string)
    else:
        fns, params, return_types = function_info
#         fns, params, return_types = extract_function_info(code1, lang)
    if len(fns) == 1:
        if lang == "Python":
            hypo_target_func = fns[0]
        else:
            hypo_target_func, target_param, target_return_type = fns[0], params[0], return_types[0]
    elif len(fns) > 1:
        max_sm = 0
        max_sm_id = 0
        for i, fn in enumerate(fns):
            if fn.lower().strip() == target_func.lower().strip():
                max_sm_id = i
                break
            sm = difflib.SequenceMatcher(None, target_func, fn).ratio()
            if sm > max_sm:
                max_sm = sm
                max_sm_id = i
        if lang == "Python":
            hypo_target_func = fns[max_sm_id]
        else:
            hypo_target_func, target_param, target_return_type \
                = fns[max_sm_id], params[max_sm_id], return_types[max_sm_id]
    return hypo_target_func, target_param, target_return_type
                         
def get_hypo_target_fn(code_string, target_func, lang):
    code1 =  get_code_for_parsing(lang, code_string)
    if lang == "Python":
        fns = get_func_names_python(code_string)
    else:
        fns, params, return_types = extract_function_info(code1, lang)
    if len(fns) == 0:
        return ""
    if len(fns) == 1:
        return fns[0]
    max_sm = 0
    max_sm_id = 0
    for i, fn in enumerate(fns):
        if fn.lower().strip() == target_func.lower().strip():
            return fn
        sm = difflib.SequenceMatcher(None, target_func, fn).ratio()
        if sm > max_sm:
            max_sm = sm
            max_sm_id = i
    return fns[max_sm_id]

def prep_exec_hypo(preds, pids, function_id_lang_dic, lang, model_type):
    programs = []
    program_id_dict = {}
    program_dict = {}
    is_plbart = False
    if model_type == "plbart":
        is_plbart = True
    for i, samples in enumerate(tqdm(preds)):
        pid = pids[i]
        program_list = get_hypo_callcode(samples, pid, lang, function_id_lang_dic, is_plbart)
        program_id_dict[pid] = [ind for ind in range(len(programs), len(programs)+len(samples))]
        program_dict[pid] = program_list
        programs += program_list
    return programs, program_id_dict, program_dict

def exec_gold(map_dict, lang, code_id_lang_dic):
    programs_gold = []
    program_id_dict_gold = {}
    pids_gold = []
    for i, pid in map_dict.items():
        programs_gold.append(code_id_lang_dic[lang][pid]['program_formatted'])
        program_id_dict_gold[pid] = [i]
        pids_gold.append(pid)
        
    lang_results_gold = p_map(file_executors[lang], programs_gold)
    result_dict_gold = {lang:lang_results_gold}
    result_type_dict_gold = show_result_summary(result_dict_gold)
    result_dicts = result_mapping(lang_results_gold, program_id_dict_gold, pids_gold, lang)
    return programs_gold, lang_results_gold, result_dicts

def get_exec_results(programs_dict):
    results_dict = {}
    for lang in programs_dict.keys():
        print(lang)
        results = p_map(file_executors[lang], programs_dict[lang])
        results_dict[lang] = results
    return results_dict

def get_exec_filtered_dict(pids_dict, result_key_lang_dict, prepro_program_dict):
    exec_pids_dict = {lang:[] for lang in langs}
    exec_prepro_program_dict = {lang:[] for lang in langs}
    for lang in langs:
        for pid, result_key, program in zip(pids_dict[lang], 
                                            result_key_lang_dict[lang], prepro_program_dict[lang]):
            if result_key == "good":
                exec_pids_dict[lang].append(pid)
                exec_prepro_program_dict[lang].append(program)
        print(lang, len(exec_pids_dict[lang]), len(result_key_lang_dict[lang])-len(exec_pids_dict[lang]))
    return exec_pids_dict, exec_prepro_program_dict

def get_len_exec_filtered_dict(pids_dict, result_key_lang_dict, length_lang_dict, prepro_program_dict):
    len_exec_pids_dict = {lang:[] for lang in langs}
    len_exec_prepro_program_dict = {lang:[] for lang in langs}
    for lang in langs:
        for pid, result_key, length, program in zip(pids_dict[lang], result_key_lang_dict[lang], 
                                           length_lang_dict[lang], prepro_program_dict[lang]):
            if result_key == 'good' and length < 510:
                len_exec_pids_dict[lang].append(pid)
                len_exec_prepro_program_dict[lang].append(program)
        print(lang, len(len_exec_pids_dict[lang]), 
              len(result_key_lang_dict[lang])-len(len_exec_pids_dict[lang]))
    return len_exec_pids_dict, len_exec_prepro_program_dict

def single_result_mapping(result_dict, pids_dict):
    result_key_lang_dict = {}
    error_type_lang_dict = {}
    for lang in result_dict.keys():
        results = result_dict[lang]
        pids = pids_dict[lang]
        result_key_lang_dict[lang] = []
        error_type_dict = {}
        for pid, result in zip(pids, results):
            result_key = result_class(result)
            result_key_lang_dict[lang].append(result_key)
            if result_key == 'error':
                error_type, error_line = error_class(result, lang)
                if error_type in error_type_dict:
                    error_type_dict[error_type] += 1
                else:
                    error_type_dict[error_type] = 1
        sorted_error_type_dict = dict(sorted(error_type_dict.items(), key=lambda item: item[1])[-10:])
        error_type_lang_dict[lang] = sorted_error_type_dict
    return result_key_lang_dict, error_type_lang_dict

def error_class(error_msg, lang="Java"):
    if lang == "C++":
        return error_class_cpp(error_msg)
    if lang == "Java":
        return error_class_cpp(error_msg, 5)
    lines = error_msg.strip().split('\n')
    if lang == "Javascript":
        for line in lines:
            if "Error:" in line:
                error_line = line
                error_type = error_line.split(':')[0]
                return error_type, error_line
    error_line = lines[-1]
    error_type = error_line.split(':')[0]
    return error_type, error_line

def error_class_cpp(error_msg, length=5):
    error_msg = error_msg.encode("ascii", "ignore").decode()
    lines = error_msg.strip().split('\n')
    for line in lines:
        if 'error:' in line:
            error_type = line.split('error:')[1]
            error_type = ' '.join(error_type.split()[:length])
            return error_type, line
    return "error", None

def result_class(result):
    result_norm = result.lower()
    error_keywords = ["error", "exception", "terminate called", "invalid", 'terminated']
    for err_key in error_keywords:
        if err_key in result_norm:
            return 'error'
    if "time out" in result_norm:
        return 'timeout'
    elif "tmp_execution_results" in result_norm:
        return 'other'
    elif len(result_norm.strip()) == 0:
        return 'empty'
    return 'good'
    
def exec_result_analysis(result_dict):
    error_dict = {}
    timeout_dict = {}
    other_dict = {}
    empty_dict = {}
    good_dict = {}
    for lang in result_dict.keys():
        results = result_dict[lang]
        if lang == 'PHP':
            results = [remove_warning_php(x) for x in result_dict[lang]]
        elif lang == 'Java':
            results = [remove_warning_java(x) for x in result_dict[lang]]
        error_list = []
        timeout_list = []
        other_list = []
        empty_list = []
        good_list = []
        for i, result in enumerate(results):
            result_key = result_class(result)
            result_norm = result.lower()
            if result_key == "error":
                error_list.append(i)
            elif result_key == "timeout":
                timeout_list.append(i)
            elif result_key == "other":
                other_list.append(i)
            elif result_key == "empty":
                empty_list.append(i)
            else:
                good_list.append(i)
        error_dict[lang] = error_list
        timeout_dict[lang] = timeout_list
        other_dict[lang] = other_list
        empty_dict[lang] = empty_list
        good_dict[lang] = good_list
    return error_dict, timeout_dict, empty_dict, other_dict, good_dict

def remove_warning_php(result):
    b = re.sub(r'(PHP Warning|PHP Notice).*line [0-9]+', '', result)
    return b

def remove_warning_java(result):
    b = re.sub(r'Note: .*\n', '', result)
    return b

def run_command(cmd, timeout=5):
    try:
        output = subprocess.run(shlex.split(cmd), 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.STDOUT, timeout=timeout).stdout.decode("unicode_escape").strip("\n")
    except subprocess.TimeoutExpired:
        output = "Time out"
        pass
    except BaseException as e:
         output = "Other error: " + str(e)
    return output

def get_file_signature():
    rand_key = random.random()
    moment = str(time.time() * 1000) + str(rand_key)
    return moment

def run_exec_java(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.java'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()
    cmd = "java " + fn_name
    output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output

def run_exec_cpp(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.cpp'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()
    exec_file = fn_name + ".out"
    cmd1 = "g++ " + fn_name + " -o " + exec_file
    compile_output = run_command(cmd1, timeout)
    os.remove(fn_name)
    if os.path.exists(exec_file):
        cmd2 = "./" + exec_file
        exec_output = run_command(cmd2, timeout)
        os.remove(exec_file)
        return exec_output
    else:
        return compile_output
    return

def run_exec_c(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.c'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()
    exec_file = fn_name + ".out"
    cmd1 = "gcc " + fn_name + " -o " + exec_file
    compile_output = run_command(cmd1, timeout)
    os.remove(fn_name)
    if os.path.exists(exec_file):
        cmd2 = "./" + exec_file
        exec_output = run_command(cmd2, timeout)
        os.remove(exec_file)
        return exec_output
    else:
        return compile_output
    return

def run_exec_csharp(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.cs'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()
    exec_file = tmp_path + moment + ".exe"
    cmd1 = "mcs " + fn_name
    compile_output = run_command(cmd1, timeout)
    os.remove(fn_name)
    if os.path.exists(exec_file):
        cmd2 = "./" + exec_file
        exec_output = run_command(cmd2, timeout)
        os.remove(exec_file)
        return exec_output
    else:
        return compile_output
    return

def run_exec_python3(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.py'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()

    cmd = "python3 " + fn_name
    output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output

def run_exec_python2(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.py'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()

    cmd = "python2 " + fn_name
    output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output

def run_exec_python(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.py'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()

    cmd = "python2 " + fn_name
    output = run_command(cmd, timeout)
    if 'error' in output:
        cmd = "python3 " + fn_name
        output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output


def run_exec_js(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.js'
    f = open(fn_name, 'w')
    new_codestring = codestring.replace("document.write", "console.log")
    f.write(new_codestring)
    f.close()

    cmd = "node " + fn_name
    output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output

def run_exec_php(codestring, timeout=2):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.php'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()

    cmd = "php " + fn_name
    output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output

def run_compile_java(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.java'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()
    cmd = "javac " + fn_name
    output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output

def run_compile_cpp(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.cpp'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()
    exec_file = fn_name + ".out"
    cmd1 = "g++ " + fn_name + " -o " + exec_file
    compile_output = run_command(cmd1, timeout)
    os.remove(fn_name)
    if os.path.exists(exec_file):
        os.remove(exec_file)
    return compile_output

def run_compile_c(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.c'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()
    exec_file = fn_name + ".out"
    cmd1 = "gcc " + fn_name + " -o " + exec_file
    compile_output = run_command(cmd1, timeout)
    os.remove(fn_name)
    if os.path.exists(exec_file):
        os.remove(exec_file)
    return compile_output

def run_compile_csharp(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.cs'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()
    exec_file = tmp_path + moment + ".exe"
    cmd1 = "mcs " + fn_name
    compile_output = run_command(cmd1, timeout)
    os.remove(fn_name)
    if os.path.exists(exec_file):
        os.remove(exec_file)
    return compile_output

def run_compile_python3(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.py'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()

    cmd = "python3 -m py_compile " + fn_name
    output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output

def run_compile_python2(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.py'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()

    cmd = "python2 -m py_compile " + fn_name
    output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output

def run_compile_python(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.py'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()

    cmd = "python3 -m py_compile " + fn_name
    output = run_command(cmd, timeout)
    if 'error' in output:
        cmd = "python2 -m py_compile " + fn_name
        output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output

def run_compile_js(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.js'
    f = open(fn_name, 'w')
    new_codestring = codestring.replace("document.write", "console.log")
    f.write(new_codestring)
    f.close()

    cmd = "node " + fn_name
    output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output

def run_compile_php(codestring, timeout=2):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.php'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()

    cmd = "php " + fn_name
    output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output



tmp_path = "./tmp_execution_results/"
file_executors = {"Java": run_exec_java, "C++": run_exec_cpp, "C": run_exec_c, "Python": run_exec_python,
                   "Javascript": run_exec_js, "PHP": run_exec_php, "C#": run_exec_csharp}
file_compilers = {"Java": run_compile_java, "C++": run_compile_cpp, "C": run_compile_c, "Python": run_compile_python, 
                  "Python3": run_compile_python3, 
                   "Javascript": run_compile_js, "PHP": run_compile_php, "C#": run_compile_csharp}