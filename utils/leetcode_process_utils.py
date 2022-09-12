from execution_utils import *
import json
import ast
import numpy as np
import re
from bs4 import BeautifulSoup
from markdown import markdown
import unicodedata

tri_langs = ['Python', 'Java', "C++"]
leetcode_pair_data_path = cwd_path + "leetcode_data_final/"
leetcode_code_dict_path = leetcode_pair_data_path + "code_dict/"

leetcode_path = "/home/mingzhu/CodeModel/CodeGen_cwd/leetcode_data/"
leetcode_data_path = leetcode_path + "intersection_data/"
python_path = leetcode_data_path + "Python/files/"
java_path = leetcode_data_path + "Java/files/"
cpp_path = leetcode_data_path + "C++/files/"
# https://github.com/lzl124631x/LeetCode
# Each problem has title, testcase, and at least 1 cpp solution
# Problems 1792
problem_path = "/home/mingzhu/CodeModel/leetcode_repos/leetcode_lzl/LeetCode/leetcode/"
# testcases from the original leetcode data we collected
testcase_path = leetcode_path + "leetcode_scraped/test_cases/processed_testcases_new_format.jsonl"

code_dict_path_new = "./leetcode_data_ming_new/"
# _readme use cpp programs from the readme files of the problem_path
code_dict_path_readme = "./leetcode_data_ming_readme/"
# This path use cpp programs from liuyubo repo
code_dict_path_liuyubo = "./leetcode_data_ming_liuyubo/"
# https://github.com/fishercoder1534/Leetcode/
# Java 1189
java_new_path = "/home/mingzhu/CodeModel/leetcode_repos/leetcode_fisher/src/main/java/com/fishercoder/solutions/"
# https://github.com/kamyu104/LeetCode-Solutions
# C++ 1710
# python 1747
cpp_new_path = "/home/mingzhu/CodeModel/leetcode_repos/leetcode_kamyu/LeetCode-Solutions/C++/"
# https://github.com/liuyubobobo/Play-Leetcode
cpp_new_path_liuyubo = "/home/mingzhu/CodeModel/leetcode_repos/leetcode_liuyubobobo/all_problems/"
# https://github.com/kamyu104/LeetCode-Solutions
python_new_path = "/home/mingzhu/CodeModel/leetcode_repos/leetcode_kamyu/LeetCode-Solutions/Python/"

lang_suffix_dict = {'Python':'.py', 'Java':'.java', "C++":".cpp"}
py_treenode_code = "class TreeNode(object):\n\tdef __init__(self, x):\n\t\tself.val = x\n\t\tself.left = None\n\t\tself.right = None\n"
py_listnode_code = "class ListNode(object):\n\tdef __init__(self, x):\n\t\tself.val = x\n\t\tself.next = None\n"

leetcode_pids_dict_path = cached_path + "leetcode_pids_dict.json"
leetcode_functions_dict_path = cached_path + "leetcode_functions_dict.json"
leetcode_functions_toked_dict_path = cached_path + "leetcode_functions_toked_dict.json"
leetcode_functions_detoked_dict_path = cached_path + "leetcode_functions_detoked_dict.json"

def get_single_functions_dict(leetcode_split_dict, leetcode_pids_dict, prepro_functions_dict):
    key_set = set()
    for tag in tags:
        key_set = key_set | set(leetcode_split_dict[tag])
    single_pids_dict = {x:[] for x in tri_langs}
    for lang, pids in leetcode_pids_dict.items():
        for pid in pids:
            if pid not in key_set:
                single_pids_dict[lang].append(pid)
        print(lang, len(single_pids_dict[lang]))
    single_functions_dict = {}
    for lang, pids in leetcode_pids_dict.items():
        single_functions_dict[lang] = []
        single_pids = set(single_pids_dict[lang])
        for i, pid in enumerate(pids):
            if pid in single_pids:
                single_functions_dict[lang].append(prepro_functions_dict[lang][i])
    return single_pids_dict, single_functions_dict

def get_md_codestring(string):
#     pattern = "(?:```[\s\S]*```)"
#     pattern = '```([^"]*)```'
    pattern = "```(.*?)```"
    reg = re.compile(pattern, re.DOTALL)
    comments = reg.findall(string)
    return comments

def add_codestring_from_readme(problems_dict):
    for key, dic in problems_dict.items():
        clean_readme = dic['clean_readme']
        sols = get_md_codestring(clean_readme)
        sol_dict = {}
        for sol in sols:
            if "class" not in sol:
                continue
            try:
                lang, new_sol = sol.strip().split('\n', 1)
                if lang in sol_dict:
                    sol_dict[lang].append(new_sol)
                else:
                    sol_dict[lang] = [new_sol]
            except:
                pass
        if len(sol_dict) == 0:
            pass
        dic['sol_dict'] = sol_dict
    return

def get_codedict(codestring_dict, lang, testcases_dict):
    code_dict = {}
    for key, codestring in codestring_dict.items():
        codestring_formatted = format_codestring(codestring, lang)
#         lang1_code, _ = remove_all_comments(codestring, lang)
#         lang1_code_format = fix_program_format(lang1_code)
#         lang1_code_enc = fix_encoding(lang1_code_format)
#         codestring_formatted = re.sub('[\n]{2,}','\n', lang1_code_enc)
        code_dict[key] = {}
        code_dict[key]['code_original'] = codestring
        code_dict[key]['code_formatted'] = codestring_formatted
        code_dict[key]['test_cases'] = {"pid":key, "tests":[]}
        if key in testcases_dict:
            code_dict[key]['test_cases'] = testcases_dict[key]
    return code_dict

def get_cpp_codedict_readme(problems_dict, testcases_dict):
    lang = "C++"
    codestring_dict = {}
    for key, dic in problems_dict.items():
        if 'cpp' not in problems_dict[key]['sol_dict']:
            continue
        codestring = problems_dict[key]['sol_dict']['cpp'][0]
        codestring_dict[key] = codestring
    code_dict = get_codedict(codestring_dict, lang, testcases_dict)
    return code_dict

def get_cpp_codedict_liuyubo(cpp_new_path_liuyubo, testcases_dict):
    lang = "C++"
    fns = os.listdir(cpp_new_path_liuyubo)
    codestring_dict = {}
    for fn in fns:
        cpp_fn_path = cpp_new_path_liuyubo + fn + '/'
        cpp_fns = os.listdir(cpp_fn_path)
        for cfn in cpp_fns:
            if cfn.startswith('cpp-'):
                _, pid = cfn.split('-')
                pid = str(int(pid))
                cf_path = cpp_fn_path + cfn + '/main.cpp'
                if os.path.isfile(cf_path):
                    with open(cf_path) as infile:
                        codestring = infile.read()
                    codestring_dict[pid] = codestring
    
    code_dict = get_codedict(codestring_dict, lang, testcases_dict)
    return code_dict



def get_problem_dict_leetcode(problem_path):
    pid_dict = {}
    problems_dict = {}
    no_readme = []
    p_fns = os.listdir(problem_path)
    for p_fn in p_fns:
        pid, title = p_fn.split('.')
        readme_path = problem_path + p_fn + '/README.md'
        if not os.path.exists(readme_path):
            no_readme.append(pid)
            continue
        problems_dict[pid] = {}
        problems_dict[pid]['idx'] = pid
        problems_dict[pid]['title'] = title.strip()
        problems_dict[pid]['readme'] = ""
        pid_dict[pid] = title.strip()
        with open(readme_path) as infile:
            problems_dict[pid]['readme'] = infile.read()
    title_pid_dict = {}
    for pid, title in pid_dict.items():
        title = title.lower().strip().replace('-', ' ')
        title = "-".join(title.split())
        if title in title_pid_dict:
            print(pid, title, title_pid_dict[title])
        title_pid_dict[title] = pid
    return problems_dict, pid_dict, title_pid_dict


def add_input_output_leetcode(problems_dict):
    for pid, problem_dic in problems_dict.items():
        readme = remove_markdown(problem_dic['readme'])
        problem_dic['clean_readme'] = readme
        lines = readme.split('\n')
        ios = {'input':[], 'output':[]}
        isstart = False
        input_idx_list = []
        output_idx_list = []
        example_idx_list = []
        emptyline_idx_list = []
        regexp = re.compile(r'Example [0-9]:')

        for i, line in enumerate(lines):
            if "Example 1:" in line:
                isstart = True
            if regexp.search(line):
                example_idx_list.append(i)
            if isstart:
                if "Input:" in line:
                    input_idx_list.append(i)
                if "Output:" in line:
                    if len(output_idx_list) >= len(input_idx_list):
                        continue
                    output_idx_list.append(i)
        for input_idx, output_idx in zip(input_idx_list, output_idx_list):
            if input_idx == output_idx:
                line_parts = lines[input_idx].split('Output:')
                ios['input'].append(line_parts[0])
                ios['output'].append('Output:' + line_parts[1])
            else:
                inp = []
                for line in lines[input_idx:output_idx]:
                    if len(line.strip()) > 0 and 'Explanation:' not in line:
                        inp.append(line)
                ios['input'].append("".join(inp))
                ios['output'].append(lines[output_idx])
        input_output_list = []
        for inp, outp in zip(ios['input'], ios['output']):
            io_dict = {}
            inp = inp.split('Input:')[1]
            outp = outp.split('Output:')[1]
            io_dict['inputs'] = "\n".join(inp.strip().split(', '))
            io_dict['output'] = outp.strip()
            input_output_list.append(io_dict)
        problem_dic['io'] = input_output_list
    return

def get_testcases_dict(testcase_path, problems_dict):
    testcases_dict = {}
    with open(testcase_path) as infile:
        lines = infile.readlines()
        for line in lines:
            dic = json.loads(line.strip())
            testcases_dict[dic['pid']] = dic
    for key, problem_dic in problems_dict.items():
        if key not in testcases_dict:
            testcases_dict[key] = {'pid':key, 'tests':problem_dic['io']}
    return testcases_dict

def get_new_leetcode_dict(lang_paths_dict, testcases_dict, title_pid_dict, problems_dict):
    lang_dict = {}
    for lang in lang_paths_dict.keys():
        codestring_dict = {}
        fns = os.listdir(lang_paths_dict[lang])
        for fn in fns:
            if not fn.endswith(lang_suffix_dict[lang]):
                continue
            code_id = fn.split('.')[0]
            if lang == "C++" or lang == "Python":
                title = fn.split('.')[0]
                if title not in title_pid_dict:
                    continue
                code_id = title_pid_dict[title.strip()]
            elif lang == "Java":
                if code_id.startswith('_') or code_id.startswith('P'):
                    code_id = code_id[1:]
            fn_path = lang_paths_dict[lang] + fn
            codestring = read_code_file(fn_path)
            codestring_dict[code_id] = codestring
            
        code_dict = get_codedict(codestring_dict, lang, testcases_dict)
        lang_dict[lang] = code_dict
    return lang_dict

def read_updated_code_dict(code_dict_path, langs=tri_langs):
    code_lang_dict = {}
    for lang in langs:
        path = code_dict_path + lang + "-code-dict-tok-updated.jsonl"
        if os.path.exists(path):
            code_dict_list = []
            with open(path) as infile:
                for line in infile:
                    code_dict_list.append(json.loads(line))
            code_lang_dict[lang] = code_dict_list
        else:
            continue
    return code_lang_dict

def read_leetcode_code_dict(code_dict_path, langs=tri_langs):
    code_lang_dict = {}
    for lang in langs:
        path = code_dict_path + lang + "-code-dict-tok.jsonl"
        if os.path.exists(path):
            code_dict_list = []
            with open(path) as infile:
                for line in infile:
                    code_dict_list.append(json.loads(line))
            code_lang_dict[lang] = code_dict_list
        else:
            continue
    return code_lang_dict

def reduce_solutions(pieces, lang):
    tok = "<entry_point>"
    program_piece = tok.join(pieces)
    program_piece = program_piece.replace(tok+target_function_place_holder+tok, tok)
    lines = program_piece.split('\n')
    solution_index_list = []
    for i, line in enumerate(lines):
        if "Solution" in line:
            solution_index_list.append(i)
    if len(solution_index_list) > 1:
        new_pieces = "\n".join(lines[:solution_index_list[1]])
        if lang == "Java":
            new_pieces += "\n}"
        return new_pieces.split(tok)
    return False

def fix_multi_solutions(code_id_lang_dict):
    bad_dict = {}
    for lang in code_id_lang_dict.keys():
        bad_list = []
        code_dict = code_id_lang_dict[lang]
        for key, code_dic in code_dict.items():
            funcs = code_dic['functions']
            if len(funcs) < 1:
                bad_list.append(key)
                continue
            func_names = code_dic['function_names']
            pieces = code_dic['program_pieces']
            new_pieces = reduce_solutions(pieces, lang)
            if new_pieces:
                code_dic['program_pieces_orignial'] = pieces
                code_dic['functions_original'] = funcs
                code_dic['program_pieces'] = new_pieces
                code_dic['functions'] = funcs[:len(new_pieces)-1]
                codestring = ""
                for piece, func in zip(new_pieces, code_dic['functions']):
                    codestring += piece
                    codestring += func
                    codestring += "\n"
                codestring += new_pieces[-1]
                code_dic['program_formatted_original'] = code_dic['program_formatted']
                code_dic['program_formatted'] = codestring
        bad_dict[lang] = bad_list
    return code_id_lang_dict

def save_code_id_dict(code_id_lang_dict, code_dict_path):
    for lang in code_id_lang_dict.keys():
        with open(code_dict_path + lang + "-code-dict-tok-updated.jsonl", 'w') as outfile:
            for key, entry in code_id_lang_dict[lang].items():
                json.dump(entry, outfile)
                outfile.write('\n')
    return

def read_code_file(fn):
    with open(fn) as infile:
        codestring = infile.read()
    return codestring

def get_id_lang_dic(code_lang_dict):
    code_id_lang_dict = {}
    for lang in code_lang_dict.keys():
        code_dicts = code_lang_dict[lang]
        code_id_dict = {}
        for code_dict in code_dicts:
            idx = code_dict["idx"].split("-")[0]
            code_id_dict[idx] = code_dict
        code_id_lang_dict[lang] = code_id_dict
    return code_id_lang_dict

def get_leetcode_dict(lang_paths_dict, testcases_dict, langs):
    lang_dict = {}
    for i, lang in enumerate(langs):
        code_dict = {}
        codestring_dict = {}
        fns = os.listdir(lang_paths_dict[lang])
        for fn in fns:
            if not fn.endswith(lang_suffix_dict[lang]):
                continue
            fn_path = lang_paths_dict[lang] + fn
            codestring = read_code_file(fn_path)
            code_id = fn.split('.')[0]
            if code_id.startswith('P'):
                code_id = code_id[1:]
            codestring_dict[code_id] = codestring
        code_dict = get_codedict(codestring_dict, lang, testcases_dict)    
        lang_dict[lang] = code_dict
    return lang_dict

def get_programs_dict(lang_dict):
    programs_dict = {}
    for lang in lang_dict.keys():
        code_dict = lang_dict[lang]
        programs_list = [v['code_formatted'] for k, v in code_dict.items()]
        programs_dict[lang] = programs_list
    return programs_dict

def save_code_dict(code_lang_dict, code_dict_path):
    for lang in code_lang_dict.keys():
        with open(code_dict_path + lang + "-code-dict.jsonl", 'w') as outfile:
            for entry in code_lang_dict[lang]:
                json.dump(entry, outfile)
                outfile.write('\n')
    return

def update_code_dict(lang_dict, code_lang_dict):
    for lang in code_lang_dict.keys():
        code_dict = lang_dict[lang]
        code_dic_list = code_lang_dict[lang]
        keys = list(code_dict.keys())
        for key, code_dic in zip(keys, code_dic_list):
            code_dic['idx'] = key
            code_dic['program_formatted'] = code_dict[key]['code_formatted']
            code_dic['test_cases'] = code_dict[key]['test_cases']
    return code_lang_dict

