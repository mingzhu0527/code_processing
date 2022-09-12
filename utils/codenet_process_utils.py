from tokenization_utils import *
from leetcode_exec_utils import *
# from tqdm.notebook import tqdm
from tqdm import tqdm



import_keyword_dict = {'Java':"import ", 'Python':"import ", 'C++':"#include", 'C':"#include", 
                          'C#':"using ", 'Javascript':"<PLACEHOLDER>", 'PHP':"<PLACEHOLDER>"}

java_imports = ['import java.awt.*;',
 'import java.io.*;',
 'import java.lang.*;',
 'import java.math.*;',
 'import java.text.*;',
 'import java.time.*;',
 'import java.util.*;',
 'import static java.lang.Double.*;',
 'import static java.lang.Integer.*;',
 'import static java.lang.Long.*;',
 'import static java.lang.Math.*;',
 'import static java.lang.StrictMath.sqrt;',
 'import static java.lang.String.*;',
 'import static java.lang.System.*;',
 'import static java.math.BigDecimal.*;',
 'import static java.math.BigInteger.*;',
 'import static java.util.Arrays.*;',
 'import static java.util.Collections.*;',
 'import static java.util.regex.Pattern.compile;']
java_imports_str = "\n".join(java_imports) + '\n'
csharp_imports_str = 'using System.Collections.Specialized;\nusing System.Text;\nusing MethodImplAttribute = System.Runtime.CompilerServices.MethodImplAttribute;\nusing System.Runtime.Remoting.Contexts;\nusing System.ComponentModel.Design;\nusing Pair = System.Collections.Generic.KeyValuePair<long, long>;\nusing MethodImplOptions = System.Runtime.CompilerServices.MethodImplOptions;\nusing System.IO.Compression;\nusing System.Text.RegularExpressions;\nusing System;\nusing System.Threading.Tasks;\nusing System.Diagnostics.Contracts;\nusing System.Runtime.CompilerServices;\nusing System.Threading;\nusing System.Web;\nusing System.Reflection;\nusing System.Diagnostics;\nusing static System.Convert;\nusing System.Net;\nusing Console = System.Console;\nusing System.Configuration;\nusing System.Security.Cryptography;\nusing System.Linq;\nusing System.Runtime.Serialization;\nusing System.Runtime.InteropServices;\nusing static System.Console;\nusing System.Globalization;\nusing System.Security.Cryptography.X509Certificates;\nusing System.Collections;\nusing System.Collections.Generic;\nusing System.IO.Pipes;\nusing System.Dynamic;\nusing static System.Math;\nusing System.IO;\n'

py_imports_str = """import sys\nimport time\nimport itertools
from itertools import accumulate, product, permutations, combinations\n
import collections
from collections import Counter, OrderedDict, deque, defaultdict, ChainMap
from functools import lru_cache\nimport math
from math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2
import fractions\nfrom typing import List, Tuple\nimport numpy as np
import random\nimport heapq\nfrom heapq import *\n"""

new_langs = ['C++', 'Java', 'Python', 'C#', 'C']

# sampled from accepted answers
codenet_data_path = "/home/mingzhu/CodeModel/Project_CodeNet/datasets/classification_data/Accepted/"
codenet_data_path_small = "/home/mingzhu/CodeModel/Project_CodeNet/datasets/codenet_accepted_small/"
codenet_io_path = "/home/mingzhu/CodeModel/Project_CodeNet/Project_CodeNet/derived/input_output/data/"
codenet_desc_path = "/home/mingzhu/CodeModel/Project_CodeNet/Project_CodeNet/problem_descriptions/"
codenet_meta_path = "/home/mingzhu/CodeModel/Project_CodeNet/Project_CodeNet/metadata/"
codenet_processed_data_path = cwd_path + "codenet_function_data/"
codenet_pair_path = codenet_processed_data_path + "codenet_function_pairs/"

def get_codenet_programs_dict(merged_filtered_dict):
    programs_dict = {}
    for lang in new_langs:
        functions = []
        for dic in merged_filtered_dict[lang]:
            functions.append(dic['function_notok'])
        programs_dict[lang] = functions
    return programs_dict

def get_merged_lang_pair_dict(lang_pair_dict):
    merged_lang_pair_dict = {}
    iterated_set = set()
    for lang1 in new_langs:
        for lang2 in new_langs:
            if lang2 == lang1:
                continue
            lang_pair1 = (lang1, lang2)
            if lang_pair1 in iterated_set:
                continue
            lang_pair2 = (lang2, lang1)
            iterated_set.add(lang_pair1)
            iterated_set.add(lang_pair2)
            src_codes1, target_codes1, pids1 = [], [], []
            src_codes2, target_codes2, pids2 = [], [], []
            if lang_pair1 in lang_pair_dict:
                src_codes1, target_codes1, pids1 = lang_pair_dict[lang_pair1]
            if lang_pair2 in lang_pair_dict:
                src_codes2, target_codes2, pids2 = lang_pair_dict[lang_pair2]
            pids = [lang1 + "-" + str(x) for x in pids1] + [lang2 + "-" + str(x) for x in pids2]
            src_codes = src_codes1 + target_codes2
            target_codes = target_codes1 + src_codes2
            merged_lang_pair_dict[lang_pair1] = [src_codes, target_codes, pids]
            print(lang_pair1, len(pids))
    return merged_lang_pair_dict


def get_all_problem_ids(merged_lang_pair_dict, merged_filtered_dict):
    all_problem_ids = set()
    for lang1, lang2 in merged_lang_pair_dict.keys():
        src_codes, target_codes, pids = merged_lang_pair_dict[(lang1, lang2)]
        for pid in pids:
            lang, ind_str = pid.split('-')
            ind = int(ind_str)
            problem_id = merged_filtered_dict[lang][ind]['pid']
            all_problem_ids.add(problem_id)
    return all_problem_ids

def get_split_lang_pair_dict(merged_lang_pair_dict, merged_filtered_dict, codenet_hypo_split_dict):
    train_set = set(codenet_hypo_split_dict['train'])
    test_set = set(codenet_hypo_split_dict['test'])
    val_set = set(codenet_hypo_split_dict['val'])
    split_lang_pair_dict = {}
    for lang1, lang2 in merged_lang_pair_dict.keys():
        print(lang1, lang2)
        split_lang_pair_dict[(lang1, lang2)] = {tag:[] for tag in tags}
        src_codes, target_codes, pids = merged_lang_pair_dict[(lang1, lang2)]
        for i, pid in enumerate(pids):
            lang, ind_str = pid.split('-')
            ind = int(ind_str)
            problem_id = merged_filtered_dict[lang][ind]['pid']
            tag = 'train'
            if problem_id in val_set:
                tag = 'val'
            elif problem_id in test_set:
                tag = 'test'
            split_lang_pair_dict[(lang1, lang2)][tag].append(i)
        print(lang1, lang2, [len(split_lang_pair_dict[(lang1, lang2)][tag]) for tag in tags])
    return split_lang_pair_dict

def write_codenet_pairdata(merged_lang_pair_dict, split_lang_pair_dict, codenet_pair_path):
    for lang1, lang2 in merged_lang_pair_dict.keys():
        src_codes, target_codes, pids = merged_lang_pair_dict[(lang1, lang2)]
        lang_pair = lang1 + "-" + lang2
        lang_pair_path = codenet_pair_path + lang_pair + '/'
        if not os.path.exists(lang_pair_path):
            os.mkdir(lang_pair_path)
        for tag in tags:
            tag_indices = split_lang_pair_dict[(lang1, lang2)][tag]
            outfile1 = open(lang_pair_path + tag + '-' + lang_pair + '-tok' + file_extensions[lang1], 'w')
            outfile2 = open(lang_pair_path + tag + '-' + lang_pair + '-tok' + file_extensions[lang2], 'w')
            outfile_map = open(lang_pair_path + tag + '-map.jsonl', 'w')
            for tag_i in tag_indices:
                outfile1.write(src_codes[tag_i] + "\n")
                outfile2.write(target_codes[tag_i] + "\n")
                outfile_map.write(pids[tag_i] + "\n")
            outfile1.close()
            outfile2.close()
            outfile_map.close()
    return

def get_lang_pair_dict(call_dict, merged_filtered_dict, programs_dict, is_plbart=False):
    lang_pair_dict = {}
    for lang1, lang2 in call_dict.keys():
        print(lang1, lang2)
        new_preds, functions, function_id_dict, call_list = call_dict[(lang1, lang2)] 
        filtered_dict = get_compiled_hypos(call_list, function_id_dict, merged_filtered_dict)
        src_codes = programs_dict[lang1]
        pids = []
        function_langs = []
        for pid, inds in filtered_dict.items():
            if len(inds) > 0:
                pids.append(pid)
                function = functions[inds[0]]
                function_langs.append((function, lang2, is_plbart))
        target_codes = p_map(notok_prepro_parallel, function_langs)
        new_pids = []
        new_src_codes = []
        new_target_codes = []
        for pid, target_code in zip(pids, target_codes):
            if target_code.strip() == "":
                continue
            new_pids.append(pid)
            new_src_codes.append(src_codes[pid])
            new_target_codes.append(target_code)
        lang_pair_list = [new_src_codes, new_target_codes, new_pids]
        lang_pair_dict[(lang1, lang2)] = lang_pair_list
    return lang_pair_dict

def get_dedup_preds(preds):
    new_preds = []
    for i, samples in enumerate(preds):
        new_preds.append(list(set(samples)))
    return new_preds

def get_hypo_callcode_codenet_target_func(samples, lang1, lang2, codenet_code_dic, is_plbart=False):
    code_dic = codenet_code_dic['code_dic']
    target_info = get_target_info(code_dic, lang1)
    print(target_info)
    if target_info == None:
        return [], []
    if lang1 == "Python":
        target_func = target_info[1]
    else:
        target_func = target_info[2]
    functions = []
    for sample in samples:
        sample = notok_detok(sample, lang2, is_plbart)
        hypo_target_info = get_hypo_target_info(sample, target_func, lang2)
        hypo_target_func = hypo_target_info[0]
        if hypo_target_func == "":
            continue
        target_hypo = sample.replace(hypo_target_func, target_func)
        functions.append(target_hypo)
    return functions

def get_target_info_set(function_names, source_return_types, source_params, lang):
    filtered_para_types = []
    filtered_return_types = []
    for func, param, return_type in zip(function_names, source_params, source_return_types):
        if func == "Main" or func == "main":
            continue
        para_list = get_para_list(param)
        if para_list == None:
            continue
        para_types = [get_real_type(x[1], lang) for x in para_list]
        filtered_para_types.append(tuple(para_types))
        filtered_return_types.append(get_real_type(return_type, lang))
    return filtered_para_types, filtered_return_types

def get_hypo_callcode_codenet(samples, lang, filtered_para_types, filtered_return_types, is_plbart=False):
    filtered_return_types_set = set(filtered_return_types)
    filtered_para_types_set = set(filtered_para_types)
    functions = []
    for sample in samples:
        target_hypo = notok_detok(sample, lang, is_plbart)
        hypo_function_info = get_hypo_function_info(target_hypo, lang)
        if lang == "Python":
            funcs = hypo_function_info
            if len(funcs) < 1:
                continue
            functions.append(target_hypo)
        else:
            funcs, params, return_types = hypo_function_info
            # Todo:
            # Check for number of parameters for python
            if len(funcs) < 1:
                continue
            if len(filtered_return_types) > 0:
                para_types, return_types = get_target_info_set(funcs, return_types, params, lang)
                for para_type, return_type in zip(para_types, return_types):
                    if return_type in filtered_return_types_set and para_type in filtered_para_types_set:
                        functions.append(target_hypo)
                        break
            else:
                functions.append(target_hypo)
    return functions


def prep_exec_hypo_codenet(preds, lang1, lang2, merged_filtered_dict, model_type):
    functions = []
    function_id_dict = {}
    is_plbart = model_type == "plbart"
    for i, samples in enumerate(tqdm(preds)):
        codenet_code_dic = merged_filtered_dict[lang1][i]
        code_dic = codenet_code_dic['code_dic']
        function_names = code_dic['function_names']
        source_return_types = code_dic['return_types']
        source_params = code_dic['parameter_lists']
#         print(source_params, source_return_types)
        filtered_para_types, filtered_return_types = get_target_info_set(
            function_names, source_return_types, source_params, lang1)
        function_list = get_hypo_callcode_codenet(samples, lang2, 
                                                  filtered_para_types, filtered_return_types, is_plbart)
#         function_list = get_hypo_callcode_codenet_target_func(samples, lang1, lang2
#         , codenet_code_dic, is_plbart)
        function_id_dict[i] = [ind for ind in range(len(functions), len(functions)+len(function_list))]
        functions += function_list
#         break
    return functions, function_id_dict

def get_hypo_call_list(functions, lang, import_str_dict):
    programs = [import_str_dict[lang] + get_code_for_parsing(lang, x) for x in functions]
    lang_results = p_map(file_compilers[lang], programs)
    _ = show_result_summary({lang:lang_results})
    processed_results, result_keys, error_type_dict = get_error_types(lang_results, lang)
    return [programs, processed_results, result_keys, error_type_dict]

def get_compiled_hypos(call_list, function_id_dict, merged_filtered_dict):
    filtered_dict = {}
    programs, processed_results, result_keys, error_type_dict = call_list
    for pid, inds in function_id_dict.items():
        filtered_dict[pid] = []
        for ind in inds:
            result_key = result_keys[ind]
            program = programs[ind]
            if result_key == "empty":
                if program.strip() != "":
                    filtered_dict[pid].append(ind)
    return filtered_dict

def get_preds_lang_dict_codenet(lang_pairs, model_type, device, programs_dict, 
                        sample_size=5, temperature=0.5,
                        data_name="xlcost", tag='test', exp_suffix="_translation_exec_function/"):
    preds_lang_dict_path = cached_path + model_type + "_" + tag + "_" + data_name + "_preds_lang_dict.pkl"
    if os.path.exists(preds_lang_dict_path):
        with open(preds_lang_dict_path, 'rb') as infile:
            preds_lang_dict = pickle.load(infile)
            return preds_lang_dict
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
        with open(preds_lang_dict_path, 'wb') as outfile:
            pickle.dump(preds_lang_dict, outfile)
    return preds_lang_dict

def get_prepro_filtered_dict(merged_filtered_dict, is_plbart=False):
    dic_path = cached_path + "codenet_merged_filtered_dict_notok.json"
    if is_plbart:
        dic_path = cached_path + "codenet_merged_filtered_dict_notok_plbart.json"
    if os.path.exists(dic_path):
        with open(dic_path, 'r') as infile:
            merged_filtered_dict = json.load(infile)
        return merged_filtered_dict
    for lang in new_langs:
        function_langs = []
        for dic in merged_filtered_dict[lang]:
            function = dic['function']
            function_langs.append((function, lang, is_plbart))
        results = p_map(notok_prepro_parallel, function_langs)
        for dic, prepro_function in zip(merged_filtered_dict[lang], results):
            dic['function_notok'] = prepro_function
    return merged_filtered_dict

def merge_filtered_dict(num_dicts):
    code_lang_dict_list = []
    call_dict_list = []
    filtered_dict_list = []
    for i in tqdm(range(num_dicts)):
        code_lang_dict_batch_path = cached_path + 'codenet_codedict_' + str(i) + '.json'
        if not os.path.exists(code_lang_dict_batch_path):
            continue
        with open(code_lang_dict_batch_path) as infile:
            code_lang_dict = json.load(infile)
        with open(cached_path + 'codenet_call_dict_' + str(i) + '.json') as infile:
            call_dict = json.load(infile)
        with open(cached_path + 'codenet_filtered_dict_' + str(i) + '.json') as infile:
            filtered_dict = json.load(infile)
        code_lang_dict_list.append(code_lang_dict)
        call_dict_list.append(call_dict)
        filtered_dict_list.append(filtered_dict)

    merged_filtered_dict = {x:[] for x in new_langs}
    for i, filtered_dict in enumerate(filtered_dict_list):
        code_lang_dict = code_lang_dict_list[i]
        for lang in new_langs:
            for fd in filtered_dict[lang]:
                fd['code_dic'] = code_lang_dict[lang][fd['code_dic_id']]
                fd['batch_id'] = i
            merged_filtered_dict[lang] += filtered_dict[lang]
    for lang in new_langs:
        print(lang, len(merged_filtered_dict[lang]))
    with open(cached_path + 'codenet_merged_filtered_dict.json', 'w') as outfile:
        json.dump(merged_filtered_dict, outfile)
    return


def split_problem_dict(codenet_problems_dict):
    problem_ids = list(codenet_problems_dict.keys())
    problem_ids.sort()
    batch_size = 100
    num_batch = len(problem_ids)//batch_size + 1
    
    for i in tqdm(range(num_batch)):
        problem_ids_seg = problem_ids[i*batch_size:(i+1)*batch_size]
        new_problem_dict = {}
        for pid in problem_ids_seg:
            new_problem_dict[pid] = codenet_problems_dict[pid]
        with open(cached_path + 'codenet_problems_dict_' + str(i) + '.json', 'w') as outfile:
            json.dump(new_problem_dict, outfile)
    return

def format_codestring_codenet(lang1_code, lang):
    lang1_code, _ = remove_all_comments(lang1_code, lang)
    lang1_code_enc = fix_encoding(lang1_code)
    codestring_formatted = re.sub('[\n]{2,}','\n', lang1_code_enc)
    return codestring_formatted


def get_codenet_programs(problems_dict, langs):
    programs_dict = {x:[] for x in langs}
    programs_id_dict = {x:[] for x in langs}
    programs_idx_dict = {x:[] for x in langs}
    for pid, problem_dict in tqdm(problems_dict.items()):
        for lang in langs:
            if lang in problem_dict['solutions']:
                programs = list(problem_dict['solutions'][lang].values())
                prepro_programs = []
                for program in programs:
                    prepro_programs.append(format_codestring_codenet(program, lang))
                programs_idx_dict[lang] += list(problem_dict['solutions'][lang].keys())
                programs_dict[lang] += prepro_programs
                programs_id_dict[lang] += [pid] * len(prepro_programs)
    return programs_dict, programs_idx_dict, programs_id_dict

def get_imports(codestring, lang):
    imports = []
    lines = codestring.split('\n')
    for line in lines:
        keywords = import_keyword_dict[lang]
        if lang == "Python":
            if line.strip().startswith(keywords) or (line.strip().startswith("from ") 
                                                     and keywords in line):
                imports.append(line.strip())
        if line.strip().startswith(keywords):
            imports.append(line.strip())
    imports_nodup = list(set(imports))
    return imports_nodup

def get_java_common_imports():
    all_imports = set()
    for lang in ['Java']:
        for i, code_dic in enumerate(codenet_codedict_small[lang]):
            code_dic = codenet_codedict_small[lang][i]
            piece = code_dic['program_pieces']
            imports = get_imports(piece[0])
            all_imports.update(imports)
    return all_imports

def get_common_imports(lang, merged_filtered_dict):
    all_imports = {}
    for i, code_dic in enumerate(merged_filtered_dict[lang]):
        code_dic = merged_filtered_dict[lang][i]['code_dic']
        piece = code_dic['program_pieces']
        imports = get_imports(piece[0], lang)
        for imp in imports:
            if imp in all_imports:
                all_imports[imp] += 1
            else:
                all_imports[imp] = 1
    sorted_all_imports = list(sorted(all_imports.items(), key=lambda item: item[1]))
    import_str = "\n".join([x[0] for x in sorted_all_imports[-50:]]) + '\n'
    return sorted_all_imports, import_str


def get_nonempty_functions(codenet_codedict_small, langs):
    func_id_dict = {}
    program_dict = {}
    imports_dict = {}
    for lang in langs:
        func_id_dict[lang] = []
        program_dict[lang] = []
        imports_dict[lang] = []
        for i, code_dic in enumerate(codenet_codedict_small[lang]):
            functions = code_dic['functions']
            function = "\n".join(functions)
            if len(functions) > 0:
                func_id_dict[lang].append(i)
                program_dict[lang].append(function)
                imports = get_imports(code_dic['program_pieces'][0], lang)
                imports_dict[lang].append("\n".join(imports) + '\n')
        print(lang, len(program_dict[lang]))
    return func_id_dict, program_dict, imports_dict

def get_compiled_functions(call_dict, func_id_dict, imports_dict, program_dict, codenet_codedict_small):
    filtered_dict = {}
    for lang in call_dict.keys():
        filtered_list = []
        programs, processed_results, result_keys, error_type_dict = call_dict[lang]
        for i, (result_key, program) in enumerate(zip(result_keys, programs)):
            if result_key == "empty":
                if program != "":
                    dic = {'code_dic_id':func_id_dict[lang][i], "import_str":imports_dict[lang][i],
                          "function":program_dict[lang][i], 
                          "pid":codenet_codedict_small[lang][func_id_dict[lang][i]]['pid']}
                    filtered_list.append(dic)
        filtered_dict[lang] = filtered_list
        print(lang, len(filtered_dict[lang]))
    return filtered_dict

def get_codenet_call_dict(program_dict, imports_dict, new_langs):
    call_dict = {}
    for lang in new_langs:
        programs = [import_str + get_code_for_parsing(lang, x)
                         for x, import_str in zip(program_dict[lang], imports_dict[lang])]
        lang_results = p_map(file_compilers[lang], programs)
        _ = show_result_summary({lang:lang_results})
        processed_results, result_keys, error_type_dict = get_error_types(lang_results, lang)
        call_dict[lang] = [programs, processed_results, result_keys, error_type_dict]
    return call_dict

def get_codenet_code_dict(programs_dict, programs_idx_dict, program_id_dict, codenet_problems_dict_batch):
    code_lang_dict = {}
    for lang in new_langs:
        code_lang_dict[lang] = []
        tuples = [(x, lang) for x in programs_dict[lang]]
        lang_dicts = p_map(get_single_code_dict, tuples)
        for i, code_dic in enumerate(lang_dicts):
            code_dic['idx'] = programs_idx_dict[lang][i]
            code_dic['pid'] = program_id_dict[lang][i]
            code_dic['program_formatted'] = programs_dict[lang][i]
            code_dic['io'] = {}
            if "io" in codenet_problems_dict_batch[program_id_dict[lang][i]]:
                code_dic['io'] = codenet_problems_dict_batch[program_id_dict[lang][i]]['io']
        code_lang_dict[lang] = lang_dicts
    return code_lang_dict


def get_codenet_dict(codenet_data_path):
    fns = os.listdir(codenet_data_path)
    problems_dict = {}
    for fn in tqdm(fns):
        problems_dict[fn] = {}
        lang_path = codenet_data_path + fn + '/'
        io_path = codenet_io_path + fn + '/'
        meta_path = codenet_meta_path + fn + '.html'
        desc_path = codenet_desc_path + fn + '.html'
        if os.path.exists(desc_path):
            with open(desc_path) as infile:
                desc = infile.read()
            problems_dict[fn]['desc'] = desc
        if os.path.exists(meta_path):
            with open(meta_path) as infile:
                meta = infile.read()
            problems_dict[fn]['meta'] = meta
        if os.path.exists(io_path):
            io_fns = os.listdir(io_path)
            problem_io_dict = {}
            for io_fn in io_fns:
                with open(io_path + '/' + io_fn) as infile:
                    io = infile.read()
                problem_io_dict[io_fn[:-4]] = io
            problems_dict[fn]['io'] = problem_io_dict

        lang_fns = os.listdir(lang_path)
        problem_dict = {}
        for lang_fn in lang_fns:
            program_path = lang_path + lang_fn + '/'
            program_fns = os.listdir(program_path)
            problem_dict[lang_fn] = {}
            for program_fn in program_fns:
                with open(program_path + '/' + program_fn) as infile:
                    program = infile.read()
                problem_dict[lang_fn][program_fn] = program
        problems_dict[fn]['solutions'] = problem_dict
    return problems_dict

