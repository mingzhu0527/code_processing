from dfg_ast_utils import *

import random

def get_exec_filtered_functions_dict(functions_pids_dict, prepro_functions_dict):
    with open(cached_path + "xlcost_exec_pids_dict.pkl", 'rb') as infile:
        exec_pids_dict = pickle.load(infile)
    exec_function_pids_dict = {lang:[] for lang in langs}
    exec_prepro_functions_dict = {lang:[] for lang in langs}
    for lang, pids in exec_pids_dict.items():
        pids_set = set(pids)
        for i, pid in enumerate(functions_pids_dict[lang]):
            if pid in pids_set:
                exec_function_pids_dict[lang].append(pid)
                exec_prepro_functions_dict[lang].append(prepro_functions_dict[lang][i])
    return exec_function_pids_dict, exec_prepro_functions_dict


def get_all_functions_detok_from_cache(function_id_lang_dic, pids_dict_path, 
                                      functions_dict_path, functions_toked_dict_path, 
                                       functions_detoked_dict_path):
    if os.path.exists(functions_dict_path) \
        and os.path.exists(functions_detoked_dict_path) \
        and os.path.exists(pids_dict_path) \
        and os.path.exists(functions_toked_dict_path):
        with open(pids_dict_path) as infile:
            pids_dict = json.load(infile)
        with open(functions_dict_path) as infile:
            functions_dict = json.load(infile)
        with open(functions_toked_dict_path) as infile:
            functions_toked_dict = json.load(infile)
        with open(functions_detoked_dict_path) as infile:
            functions_detoked_dict = json.load(infile)
    else:
        pids_dict = {}
        functions_dict = {}
        functions_toked_dict = {}
        functions_detoked_dict = {}
        for lang, lang_dic in function_id_lang_dic.items():
            functions = []
            toked_functions = []
            detoked_functions = []
            pids = list(lang_dic.keys())
            for pid in tqdm(pids):
                v = lang_dic[pid]
                function = "\n".join(v['functions'])
                if lang == "Python":
                    join_str = ' NEW_LINE '
                    function_tokenized = join_str.join([" ".join(tokens) for tokens in v['tokens']])
                    toked_functions.append(function_tokenized)
                    function_detok = detok_format(function_tokenized, file_detokenizers[lang])
                    detoked_functions.append(function_detok)
                functions.append(function)
            functions_dict[lang] = functions
            if lang == "Python":
                functions_toked_dict[lang] = toked_functions
                functions_detoked_dict[lang] = detoked_functions
            pids_dict[lang] = pids
        with open(pids_dict_path, 'w') as outfile:
            json.dump(pids_dict, outfile)
        with open(functions_dict_path, 'w') as outfile:
            json.dump(functions_dict, outfile)
        with open(functions_toked_dict_path, 'w') as outfile:
            json.dump(functions_toked_dict, outfile)
        with open(functions_detoked_dict_path, 'w') as outfile:
            json.dump(functions_detoked_dict, outfile)
    return pids_dict, functions_dict, functions_toked_dict, functions_detoked_dict

def read_function_tok_file_filtered():
    """
    this is for stats collection. Thus remove the empty lines.
    """
    id_lang_dic = {}
    json_dict = {}
    for lang in file_tokenizers.keys():
        with open(function_map_data_path + lang + "-program-functions.jsonl") as infile:
            lines = infile.readlines()
            jsons = [json.loads(line.strip()) for line in lines]
            json_dict[lang] = []
            id_dic = {}
            program_dic = {}
            for js in jsons:
                if js['bpe'] == "":
                    continue
                if "" in js['code_dict']['bpe']:
                    continue
                idx_l = js['idx'].split('-')
                pid = idx_l[0]
                id_dic[pid] = js
                json_dict[lang].append(js)
            id_lang_dic[lang] = id_dic
    return json_dict, id_lang_dic

def read_function_tok_file():
    id_lang_dic = {}
    json_dict = {}
    for lang in file_tokenizers.keys():
        with open(function_map_data_path + lang + "-program-functions.jsonl") as infile:
#         with open(code_dict_path + lang + "-code-dict-tok.jsonl") as infile:
            lines = infile.readlines()
            jsons = [json.loads(line.strip()) for line in lines]
            json_dict[lang] = jsons
            id_dic = {}
            program_dic = {}
            for js in jsons:
                idx_l = js['idx'].split('-')
                pid = idx_l[0]
                id_dic[pid] = js
            id_lang_dic[lang] = id_dic
    return json_dict, id_lang_dic

def get_pairwise_functions(pair_path, test_list, val_list, program_id_lang_dic):
    pair_path += "pair_data_tok_function/"
    if not os.path.exists(pair_path):
        os.makedirs(pair_path)

    iterated_set = set()
    test_set = set(test_list)
    val_set = set(val_list)
    all_set = test_set | val_set
    problem_desc_dic = {}
    for lang1 in file_tokenizers.keys():
        for lang2 in file_tokenizers.keys():
            if lang2 == lang1:
                continue
            if (lang2, lang1) in iterated_set:
                continue
            iterated_set.add((lang1, lang2))
            print(lang1, lang2)
            counter_dict = {tag: 0 for tag in tags}
            file_handler_dic = get_fhs(pair_path, lang1, lang2, 'full')
            
            for k, js1 in program_id_lang_dic[lang1].items():
                if k in program_id_lang_dic[lang2]:
                    if k not in all_set:
                        tag = 'train'
                    elif k in test_set:
                        tag = 'test'
                    elif k in val_set:
                        tag = 'val'
                    js2 = program_id_lang_dic[lang2][k]
                    
                    # TODO
                    # use correct new_line to connect different functions
                    bpe_l1 = js1['code_dict']['bpe']
                    bpe_l2 = js2['code_dict']['bpe']
                    tok_l1 = js1['code_dict']['tokens']
                    tok_l2 = js2['code_dict']['tokens']
                    if len(bpe_l1) == 0 or len(bpe_l2) == 0:
                        continue
#                     if '' in bpe_l1 or '' in bpe_l2:
#                         continue
                    
                    join_str1 = " "
                    join_str2 = " "
                    if lang1 == "Python":
                        join_str1 = " NEW_LINE "
                    if lang2 == "Python":
                        join_str2 = " NEW_LINE "
                    bpe_str1 = join_str1.join(bpe_l1)
                    bpe_str2 = join_str2.join(bpe_l2)
                    tok_str1 = join_str1.join([" ".join(tokens) for tokens in tok_l1])
                    tok_str2 = join_str2.join([" ".join(tokens) for tokens in tok_l2])
                    counter_dict[tag] += 1
                    trans1 = ""
                    trans2 = ""
                    
                    if tag != 'train':
                        pdesc = js1['problem_desc']
                        if pdesc in problem_desc_dic:
                            pdtok = problem_desc_dic[pdesc]
                        else:
                            pdtok = java_tokenizer(pdesc)
                            pdtok_bpe = bpe_model.apply_bpe(" ".join(pdtok))
                            problem_desc_dic[pdesc] = pdtok_bpe
                        sid_tok_bpe = pdtok_bpe + ' ' + str(k)
                        trans1 = sid_tok_bpe + " | " + bpe_str1 
                        trans2 = sid_tok_bpe + " | " + bpe_str2
                        
                    write_files(tag, bpe_str1, bpe_str2,
                                tok_str1, tok_str2,
                                k, k,
                                trans1, trans2,
                                file_handler_dic)      
            for tag in tags:
                for fh in file_handler_dic[tag]:
                    fh.close()
                print(tag, counter_dict[tag])
    return

def get_toked_functions(jsons, tokz, key='functions'):
    failed = []
    for i, js in enumerate(tqdm(jsons)):
        functions = js[key]
        js['tokens'] = []
        js['bpe'] = []
        for function in functions:
            function = function.replace('\\n', '\n').replace('\\t', '\t')
            try:
                tokens = tokz(function)
                tokens_bpe = bpe_model.apply_bpe(" ".join(tokens))
                js['tokens'].append(tokens)
                js['bpe'].append(tokens_bpe)
            except:
                failed.append(js['idx'])
                js['tokens'].append("")
                js['bpe'].append("")
    return failed

def tokenize_functions(file_suffix, key, code_dict_path, langs=langs):
    json_dict = {}
    fail_dict = {}
    for lang in langs:
        tokz = file_tokenizers[lang]
        with open(code_dict_path + lang + "-" + file_suffix + ".jsonl") as infile:
            lines = infile.readlines()
            jsons = [json.loads(line.strip()) for line in lines]
            failed_list = get_toked_functions(jsons, tokz, key) #program_formatted program
            fail_dict[lang] = failed_list
            json_dict[lang] = jsons

    # Print tokenization failed cased
    for lang in langs:
        print(lang, len(fail_dict[lang]))
        print(lang, len(json_dict[lang]))

    # Save tokenized mapping files
    for lang in langs:
        jsons = json_dict[lang]
        with jsonlines.open(code_dict_path + lang + "-" + file_suffix + "-tok.jsonl", 'w') as outfile:
            outfile.write_all(jsons)
    return

def get_code_for_parsing(lang, func):
    code1 = func
    if lang == 'Java':
        code1 = "class G4G{\n" + func + "\n}"
    elif lang == 'C#':
        code1 = "class G4G{\n" + func + "\nstatic void Main(){}}"
    elif lang == 'C++':
        code1 = "using namespace std;\n" + func + "\nint main(){return 0;}"
    elif lang == 'C':
        code1 = func + "\nint main(){return 0;}"
    elif lang == "PHP":
        code1 = "<?php \n" + func + "\n?>"
    return code1

def check_cross_call(call_dict, function_names):
    black_set = set()
    for key, calls_set in call_dict.items():
        for fn in function_names:
            if fn != key:
                if fn in calls_set:
                    black_set.add(fn)
    target_fns = []
    for fn in function_names:
        if fn not in black_set:
            target_fns.append(fn)
    return target_fns

def remove_print_func(target_funcs):
    target_funcs_new = []
    for func in target_funcs:
        if 'print' not in func.lower():
            target_funcs_new.append(func)
    return target_funcs_new

def remove_cross_call(lang, functions, function_names):
    call_dict = {}
    for func, fn in zip(functions, function_names):
        func_code1 = get_code_for_parsing(lang, func)
        calls, call_args, call_arg_dict = extract_function_call_info(func_code1, lang)
        call_dict[fn] = set(calls)
    target_fns = check_cross_call(call_dict, function_names)
    return target_fns

def save_function_map_data(code_lang_dict, programs_dict, program_id_lang_dic):
    for lang in langs:
        keys = program_id_lang_dic[lang].keys()
        for i, key in enumerate(keys):
            v = program_id_lang_dic[lang][key]
            program = programs_dict[lang][i]
            code_dict = code_lang_dict[lang][i]
            code_dict['idx'] = v['idx']
            assert(program == v['program_formatted'])
            v['code_dict'] = code_dict

        dic = program_id_lang_dic[lang]
        vs = list(dic.values())
        with open(function_map_data_path + lang + "-program-functions.jsonl", 'w') as outfile:
            for entry in vs:
                json.dump(entry, outfile)
                outfile.write('\n')
    return

def update_code_dict(code_lang_dict, programs_dict, program_id_lang_dic):
    for lang in langs:
        keys = program_id_lang_dic[lang].keys()
        for i, key in enumerate(keys):
            v = program_id_lang_dic[lang][key]
            program = programs_dict[lang][i]
            code_lang_dict[lang][i]['idx'] = v['idx']
            code_lang_dict[lang][i]['program_formatted'] = v['program_formatted']
            assert(program == v['program_formatted'])
    return code_lang_dict

def save_code_dict(code_lang_dict):
    if not os.path.exists(code_dict_path):
        os.mkdir(code_dict_path)
    for lang in code_lang_dict.keys():
        with open(code_dict_path + lang + "-code-dict.jsonl", 'w') as outfile:
            for entry in code_lang_dict[lang]:
                json.dump(entry, outfile)
                outfile.write('\n')
    return

def read_function_tok_file1():
    id_lang_dic = {}
    json_dict = {}
    for lang in file_tokenizers.keys():
        with open(function_map_data_path + lang + "-program-functions.jsonl") as infile:
#         with open(code_dict_path + lang + "-code-dict-tok.jsonl") as infile:
            lines = infile.readlines()
            jsons = [json.loads(line.strip()) for line in lines]
            json_dict[lang] = jsons
            id_dic = {}
            program_dic = {}
            for js in jsons:
                idx_l = js['idx'].split('-')
                pid = idx_l[0]
                id_dic[pid] = js
            id_lang_dic[lang] = id_dic
    return json_dict, id_lang_dic

def read_toked_code_dict(code_dict_path, langs=langs):
    code_lang_dict = {}
    code_id_lang_dic = {}
    for lang in langs:
        path = code_dict_path + lang + "-code-dict-tok.jsonl"
        if not os.path.exists(path):
            path = code_dict_path + lang + "-code-dict.jsonl"
        code_dict_list = []
        id_dic = {}
        with open(path) as infile:
            for line in infile:
                js = json.loads(line)
                idx_l = js['idx'].split('-')
                pid = idx_l[0]
                id_dic[pid] = js
                code_dict_list.append(js)
        code_lang_dict[lang] = code_dict_list
        code_id_lang_dic[lang] = id_dic
    return code_lang_dict, code_id_lang_dic

def read_code_dict(programs_dict, program_id_lang_dic):
    code_lang_dict = {}
    for lang in langs:
        path = code_dict_path + lang + "-code-dict.jsonl"
        if os.path.exists(path):
            code_dict_list = []
            with open(path) as infile:
                for line in infile:
                    code_dict_list.append(json.loads(line))
            code_lang_dict[lang] = code_dict_list
        else:
            break
    if code_lang_dict == {}:
        code_lang_dict, empty_target_dict = get_code_dict(programs_dict)
        code_lang_dict = update_code_dict(code_lang_dict, programs_dict, program_id_lang_dic)
        save_code_dict(code_lang_dict)
    return code_lang_dict

def get_func_info(v, graph1, lang):
    children = v['children']
    func_name = ""
    param_list = ""
    return_type = ""
    if lang == "Java":
        for i, child in enumerate(children):
            if child.startswith("identifier"):
                return_type = graph1[children[i-1]]['snippet']
                func_name = graph1[child]['snippet']
            if child.startswith("formal_parameters"):
                param_list = graph1[child]['snippet']
    elif lang == "Python":
        func_key = children[1]
        func_name = graph1[func_key]['snippet']
        for child in children:
            if child.startswith("parameters"):
                param_list = graph1[child]['snippet']
    elif lang == "Javascript" or lang == "PHP":
        func_key = children[1]
        func_name = graph1[func_key]['snippet']
        for child in children:
            if child.startswith("formal_parameters"):
                param_list = graph1[child]['snippet']
    elif lang == "C#":
        for i, child in enumerate(children):
            if child.startswith("parameter_list"):
                param_list = graph1[child]['snippet']
                prev_child = children[i-1]
                if prev_child.startswith("identifier"):
                    return_type = graph1[children[i-2]]['snippet']
                    func_name = graph1[prev_child]['snippet']
    elif lang == "C" or lang == "C++":
        return_type_child = children[0]
        grandchildren = graph1[return_type_child]['children']
        if len(grandchildren) == 0:
            return_type = graph1[return_type_child]['snippet']
        else:
            snippets = []
            for gc in grandchildren:
                snippets.append(graph1[gc]['snippet'])
            return_type = "".join(snippets)
        for child in children:
#             if child.startswith("function_definition"):
#                 grandchildren = graph1[child]['children']
#                 return_type = graph1[grandchildren[0]]['snippet']
            if child.startswith("function_declarator"):
                grandchildren = graph1[child]['children']
                func_name = graph1[grandchildren[0]]['snippet']
                for grandchild in grandchildren:
                    if grandchild.startswith("parameter_list"):
                        param_list = graph1[grandchild]['snippet']
        if func_name == "":
            for child in children:
                if child.startswith("pointer_declarator"):
                    return_type += "*"
                    for grandchild in graph1[child]['children']:
                        if grandchild.startswith("function_declarator"):
                            greatgrandchildren = graph1[grandchild]['children']
                            func_name = graph1[greatgrandchildren[0]]['snippet']
                            for greatgrandchild in greatgrandchildren:
                                if greatgrandchild.startswith("parameter_list"):
                                    param_list = graph1[greatgrandchild]['snippet']
                                    break
                            break
                    break
    return func_name, param_list, return_type

def get_func_info_old(v, graph1, lang):
    children = v['children']
    func_name = ""
    param_list = ""
    return_type = ""
    if lang == "Java":
        for i, child in enumerate(children):
            if child.startswith("identifier"):
                return_type = graph1[children[i-1]]['snippet']
                func_name = graph1[child]['snippet']
            if child.startswith("formal_parameters"):
                param_list = graph1[child]['snippet']
    elif lang == "Python":
        func_key = children[1]
        func_name = graph1[func_key]['snippet']
        for child in children:
            if child.startswith("parameters"):
                param_list = graph1[child]['snippet']
    elif lang == "Javascript" or lang == "PHP":
        func_key = children[1]
        func_name = graph1[func_key]['snippet']
        for child in children:
            if child.startswith("formal_parameters"):
                param_list = graph1[child]['snippet']
    elif lang == "C#":
        for i, child in enumerate(children):
            if child.startswith("parameter_list"):
                param_list = graph1[child]['snippet']
                prev_child = children[i-1]
                if prev_child.startswith("identifier"):
                    return_type = graph1[children[i-2]]['snippet']
                    func_name = graph1[prev_child]['snippet']
    elif lang == "C" or lang == "C++":
        for child in children:
            if child.startswith("function_definition"):
                grandchildren = graph1[child]['children']
                return_type = graph1[grandchildren[0]]['snippet']
            if child.startswith("function_declarator"):
                grandchildren = graph1[child]['children']
                func_name = graph1[grandchildren[0]]['snippet']
                for grandchild in grandchildren:
                    if grandchild.startswith("parameter_list"):
                        param_list = graph1[grandchild]['snippet']
        if func_name == "":
            for child in children:
                if child.startswith("pointer_declarator"):
                    for grandchild in graph1[child]['children']:
                        if grandchild.startswith("function_declarator"):
                            greatgrandchildren = graph1[grandchild]['children']
                            func_name = graph1[greatgrandchildren[0]]['snippet']
                            for greatgrandchild in greatgrandchildren:
                                if greatgrandchild.startswith("parameter_list"):
                                    param_list = graph1[greatgrandchild]['snippet']
    return func_name, param_list, return_type

def extract_function_info(code1, lang):
    root_node1 = get_ast(code1, ast_parsers[lang])
    root1, graph1 = get_graph(code1, root_node1, lang)
    fn_list = []
    params = []
    return_types = []
    fn_param_dict = {}
    for k, v in graph1.items():
        function_keyword = function_dict[lang]
        if k.startswith(function_keyword):
            func_name, param, return_type = get_func_info(v, graph1, lang)
            func_name = func_name.strip()
            param = param.strip()
            return_type = return_type.strip()
            fn_list.append(func_name)
            params.append(param)
            return_types.append(return_type)
    return fn_list, params, return_types

def random_sample_programs_new():
    error_dict = {}
    error_dict_finer = {}
    function_info_dict = {}
    for lang in langs:
        error_count = 0
        error_list = []
        error_list_finer = []
        function_info = []
        for key in tqdm(range(len(programs_dict[lang]))):
            code1 = programs_dict[lang][key]
            functions, program_pieces = separate_functions_from_programs(code1, lang)
            func_names, params = extract_function_info(code1, lang)
            function_info.append((func_names, params))
            for func_name, param in zip(func_names, params):
                if func_name == "" or param == "":
                    error_list_finer.append(key)
            if len(func_names) == 0:
                error_list.append(key)
                error_count += 1
            program_piece = " ".join(program_pieces)
            func_call_names = extract_function_calls(program_piece, lang)
        print(lang, error_count, len(error_list_finer))
        error_dict[lang] = error_list
        error_dict_finer[lang] = error_list_finer
        function_info_dict[lang] = function_info
    return error_dict, error_dict_finer, function_info_dict

def add_target_function_placeholder(program_pieces, lang):
    insert_position = 1
    if len(program_pieces) == 0 or len(program_pieces) == 1:
        insert_position = 0
    program_pieces.insert(insert_position, target_function_place_holder)
    return program_pieces

def remove_undefined_call(func_call_names, func_names):
    func_names_set = set(func_names)
    target_funcs = []
    for func_call in func_call_names:
        if func_call in func_names_set:
            target_funcs.append(func_call)
    return target_funcs

def remove_system_call(target_funcs, sys_calls_dict, lang):
    if len(target_funcs) == 0:
        return target_funcs
    target_funcs_new = []
    for func in target_funcs:
        if func not in sys_calls_dict[lang]:
            target_funcs_new.append(func)
    return target_funcs_new



def check_target_call(target_funcs, func_call_names):
    if target_funcs == 1:
        return True, target_funcs[0]
    if len(target_funcs) == 0:
        target_funcs = func_call_names
    return False, target_funcs



def get_target_call_old(func_call_names, func_names, lang):
    
    func_call_names = list(dict.fromkeys(func_call_names)) 
    func_names_set = set(func_names)
    target_funcs = []
    for func_call in func_call_names:
        if func_call in func_names_set:
            target_funcs.append(func_call)
    target_funcs = list(dict.fromkeys(target_funcs)) 
    
    if len(target_funcs) == 0:
        target_funcs = func_call_names
    elif len(target_funcs) > 1:
        target_funcs_new = []
        for func in target_funcs:
            if func not in sys_calls_dict[lang]:
                target_funcs_new.append(func)
        if len(target_funcs_new) == 0:
            target_funcs = func_call_names
        target_funcs = target_funcs_new
    
    target_func = ""
    if len(target_funcs) > 1:
        for func in target_funcs:
            if 'print' not in func.lower():
                target_func = func
        if target_func == "":
            target_func = target_funcs[-1]
    elif len(target_funcs) == 1:
        target_func = target_funcs[0]
    return target_func

def extract_function_call_info_old(code1, lang):
    root_node1 = get_ast(code1, ast_parsers[lang])
    root1, graph1 = get_graph(code1, root_node1, lang)
    fn_list = []
    params = []
    fn_param_dict = {}
    for k, v in graph1.items():
        function_call_keyword = function_call_dict[lang]
        if k.startswith(function_call_keyword):
            children = v['children']
            func_name = graph1[children[0]]['snippet']
            param = graph1[children[1]]['snippet']
            func_name = func_name.strip()
            param = param.strip()
            fn_list.append(func_name)
            params.append(param)
            if func_name not in fn_param_dict:
                fn_param_dict[func_name] = param
    return fn_list, params, fn_param_dict

def extract_function_call_info(code1, lang):
    root_node1 = get_ast(code1, ast_parsers[lang])
    root1, graph1 = get_graph(code1, root_node1, lang)
    fn_list = []
    params = []
    fn_param_dict = {}
    for k, v in graph1.items():
        function_call_keyword = function_call_dict[lang]
        if k.startswith(function_call_keyword):
            children = v['children']
            # TODO
            # if it's a system call, should we keep the object or not?
            func_name = graph1[children[-2]]['snippet']
            param = graph1[children[-1]]['snippet']
            func_name = func_name.strip()
            param = param.strip()
            fn_list.append(func_name)
            params.append(param)
            if func_name not in fn_param_dict:
                fn_param_dict[func_name] = param
    return fn_list, params, fn_param_dict

def get_target_call(func_call_names, func_names, sys_calls_dict, lang):
    if len(func_call_names) == 0 or len(func_names) == 0:
        return ""
    func_call_names = list(dict.fromkeys(func_call_names)) 
        
    target_funcs = remove_undefined_call(func_call_names, func_names)
    if target_funcs == 1:
        return target_funcs[0]
    if len(target_funcs) == 0:
        target_funcs = func_call_names
    
    target_funcs = remove_system_call(target_funcs, sys_calls_dict, lang)
    if target_funcs == 1:
        return target_funcs[0]
    if len(target_funcs) == 0:
        return ""
    
    target_funcs = remove_print_func(target_funcs)
    if target_funcs == 1:
        return target_funcs[0]
    if len(target_funcs) == 0:
        return ""

    return target_funcs[0]

def get_target_fn(funcs, lang, func_call_names):
    # 不存在undefined call的情况
    # 不存在system call的情况
    # 可能存在print的情况
    code1 =  get_code_for_parsing(lang, "\n".join(funcs))
    fns, params, return_types = extract_function_info(code1, lang)
    if len(fns) == 0:
        return ""
    if len(fns) == 1:
        return fns[0]
    target_fns = remove_cross_call(lang, funcs, fns)
    if len(target_fns) == 0:
        return ""
    if len(target_fns) == 1:
        return target_fns[0]
    
    func_calls = set(func_call_names)
    target_fn_called = []
    for target_fn in target_fns:
        if target_fn in func_calls:
            target_fn_called.append(target_fn)
    if len(target_fn_called) == 0:
        return ""
    if len(target_fn_called) == 1:
        return target_fn_called[0]
    
    target_fns_print = remove_print_func(target_fn_called)
    if len(target_fns_print) > 0:
        return target_fns_print[0]
    return target_fns[0]

def get_single_code_dict(code_lang):
    code1, lang = code_lang
    code_dict = {}
    functions, program_pieces = separate_functions_from_programs(code1, lang)
    program_pieces = add_target_function_placeholder(program_pieces, lang)
    func_names, params, return_types = extract_function_info(code1, lang)
    program_piece = " ".join(program_pieces).replace(target_function_place_holder, "")
    func_call_names, func_call_params, fn_param_call_dict = extract_function_call_info(
        program_piece, lang)
#             target_func = get_target_call(func_call_names, func_names, sys_calls_dict, lang)
    target_func = get_target_fn(functions, lang, func_call_names)

    code_dict['functions'] = functions
    code_dict['program_pieces'] = program_pieces
    code_dict['function_names'] = func_names
    code_dict['parameter_lists'] = params
    code_dict['return_types'] = return_types
    code_dict['target_call'] = target_func
    # check the mapping to the original param_list
    code_dict['target_call_args'] = ""
    code_dict['target_call_params'] = ""
    code_dict['target_call_return_type'] = ""
    if target_func != "":
        if target_func in fn_param_call_dict:
            code_dict['target_call_args'] = fn_param_call_dict[target_func]
        for x, y, z in zip(func_names, params, return_types):
            if target_func == x:
                code_dict['target_call_params'] = y
                code_dict['target_call_return_type'] = z
                break
    return code_dict

def get_code_dict(programs_dict):
    empty_target_dict = {}
    code_lang_dict = {}
    for lang in programs_dict.keys():
        print(lang)
#         for escape_call in escape_list:
#             sys_calls_dict[lang][escape_call] = 1
        code_lang_list = []
        empty_target_list = []
        for i, code1 in enumerate(tqdm(programs_dict[lang])):
            code_dict = get_single_code_dict(code1, lang)
#             code_dict = {}
#             functions, program_pieces = separate_functions_from_programs(code1, lang)
#             program_pieces = add_target_function_placeholder(program_pieces, lang)
#             func_names, params, return_types = extract_function_info(code1, lang)
            
#             program_piece = " ".join(program_pieces).replace(target_function_place_holder, "")
#             func_call_names, func_call_params, fn_param_call_dict = extract_function_call_info(
#                 program_piece, lang)
# #             target_func = get_target_call(func_call_names, func_names, sys_calls_dict, lang)
#             target_func = get_target_fn(functions, lang, func_call_names)
            
#             code_dict['functions'] = functions
#             code_dict['program_pieces'] = program_pieces
#             code_dict['function_names'] = func_names
#             code_dict['parameter_lists'] = params
#             code_dict['return_types'] = return_types
#             code_dict['target_call'] = target_func
#             # check the mapping to the original param_list
#             code_dict['target_call_args'] = ""
#             code_dict['target_call_params'] = ""
#             code_dict['target_call_return_type'] = ""
#             if target_func == "":
#                 empty_target_list.append(i)
#             else:
#                 if target_func in fn_param_call_dict:
#                     code_dict['target_call_args'] = fn_param_call_dict[target_func]
#                 else:
#                     print(lang, i, target_func, fn_param_call_dict)
#                 for x, y, z in zip(func_names, params, return_types):
#                     if target_func == x:
#                         code_dict['target_call_params'] = y
#                         code_dict['target_call_return_type'] = z
#                         break
            code_lang_list.append(code_dict)
        empty_target_dict[lang] = empty_target_list
        code_lang_dict[lang] = code_lang_list
    return code_lang_dict, empty_target_dict

def get_param_list(v, graph1, lang):
    children = v['children']
    param_list = ""
    if lang == "Java":
        for child in children:
            if child.startswith("formal_parameters"):
                param_list = graph1[child]['snippet']
    elif lang == "C++":
         for child in children:
            if child.startswith("function_declarator"):
                grandchildren = graph1[child]['children']
                for grandchild in grandchildren:
                    if grandchild.startswith("parameter_list"):
                        param_list = graph1[grandchild]['snippet']
    elif lang == "Python":
        for child in children:
            if child.startswith("parameters"):
                param_list = graph1[child]['snippet']
    elif lang == "Javascript":
        for child in children:
            if child.startswith("formal_parameters"):
                param_list = graph1[child]['snippet']
    elif lang == "C#":
        for i, child in enumerate(children):
            if child.startswith("parameter_list"):
                param_list = graph1[child]['snippet']
    elif lang == "C":
        for child in children:
            if child.startswith("function_declarator"):
                grandchildren = graph1[child]['children']
                for grandchild in grandchildren:
                    if grandchild.startswith("parameter_list"):
                        param_list = graph1[grandchild]['snippet']
    return param_list

def get_func_name(v, graph1, lang):
    children = v['children']
    func_name = ""
    if lang == "Java":
        for child in children:
            if child.startswith("identifier"):
                func_name = graph1[child]['snippet']
    elif lang == "C++":
         for child in children:
            if child.startswith("function_declarator"):
                func_key = graph1[child]['children'][0]
                func_name = graph1[func_key]['snippet']
    elif lang == "Python" or lang == "Javascript":
        func_key = children[1]
        func_name = graph1[func_key]['snippet']
    elif lang == "C#":
        for i, child in enumerate(children):
            if child.startswith("parameter_list"):
                prev_child = children[i-1]
                if prev_child.startswith("identifier"):
                    func_name = graph1[prev_child]['snippet']
    elif lang == "C":
        for child in children:
            if child.startswith("function_declarator"):
                func_key = graph1[child]['children'][0]
                func_name = graph1[func_key]['snippet']
        if func_name == "":
            for child in children:
                if child.startswith("pointer_declarator"):
                    for grandchild in graph1[child]['children']:
                        if grandchild.startswith("function_declarator"):
                            func_key = graph1[grandchild]['children'][0]
                            func_name = graph1[func_key]['snippet']
    return func_name

def extract_function_names(code1, lang):
    root_node1 = get_ast(code1, ast_parsers[lang])
    root1, graph1 = get_graph(code1, root_node1, lang)
    fn_list = []
    for k, v in graph1.items():
        function_keyword = function_dict[lang]
        if k.startswith(function_keyword):
            func_name = get_func_name(v, graph1, lang)
            fn_list.append(func_name.strip())
    return fn_list

def extract_function_calls(code1, lang):
    root_node1 = get_ast(code1, ast_parsers[lang])
    root1, graph1 = get_graph(code1, root_node1, lang)
    fn_list = []
    for k, v in graph1.items():
        function_call_keyword = function_call_dict[lang]
        if k.startswith(function_call_keyword):
            children = v['children']
            func_name = graph1[children[0]]['snippet']
            fn_list.append(func_name.strip())
    return fn_list

def get_function_indexes(code1, lang):
    root_node1 = get_ast(code1, ast_parsers[lang])
    root1, graph1 = get_graph(code1, root_node1, lang)
    function_index_list = []
    for k, v in graph1.items():
        function_keyword = function_dict[lang]
        if k.startswith(function_keyword):
            func_name = get_func_name(v, graph1, lang)
            if func_name.lower().strip() != "main":
                function_index_list.append(v['start_end'])
    if len(function_index_list) == 0:
        return [], [(0, len(code1))]
    program_index_list = []
    for start, end in function_index_list:
        if len(program_index_list) == 0:
            program_index_list.append((0, start))
        else:
            s, e = program_index_list[-1]
            program_index_list[-1] = (s, start)
        program_index_list.append((end, -1))
    s, e = program_index_list[-1]
    program_index_list[-1] = (s, len(code1))

    for (s1, e1), (s2, e2) in zip(program_index_list, function_index_list):
        assert(e1==s2)
    return function_index_list, program_index_list

def separate_functions_from_programs(code1, lang):
    function_index_list, program_index_list = get_function_indexes(code1, lang)
    functions = []
    program_pieces = []
    for start, end in function_index_list:
        function = code1[start:end]
        functions.append(function)
    for start, end in program_index_list:
        program_piece = code1[start:end]
        program_pieces.append(program_piece)
    return functions, program_pieces

def get_sys_calls(sys_calls_path, programs_dict):
    if os.path.exists(sys_calls_path):
        with open(sys_calls_path) as infile:
            sys_calls_dict = json.load(infile)
        return sys_calls_dict
    system_call_dict = {}
    target_call_dict = {}
    for lang in langs:
        system_calls = dict()
        target_calls = dict()
        for i in tqdm(range(len(programs_dict[lang]))):
            code1 = programs_dict[lang][i]
            functions, program_pieces = separate_functions_from_programs(code1, lang)
            func_names = extract_function_names(code1, lang)
            program_piece = " ".join(program_pieces)
            func_call_names = extract_function_calls(program_piece, lang)
            func_names_set = set(func_names)

            func_calls = []
            sys_calls = []
            for func_call in func_call_names:
                if func_call not in func_names_set:
                    sys_calls.append(func_call)
            system_calls[i] = sys_calls
            target_calls[i] = func_names
    #         break
        system_call_dict[lang] = system_calls
        target_call_dict[lang] = target_calls
    sys_calls_dict = {}
    for lang, v in system_call_dict.items():
        sys_calls = {}
        for k, calls in v.items():
            for call in calls:
                if call in sys_calls:
                    sys_calls[call] += 1
                else:
                    sys_calls[call] = 1
        sys_calls_dict[lang] = sys_calls
    with open(sys_calls_path, 'w') as outfile:
        json.dump(sys_calls_dict, outfile)
    return sys_calls_dict
#         sorted_calls = list(sorted(sys_calls.items(), key=lambda x: x[1]))
#         print(lang, len(sorted_calls), sorted_calls[-10:])

def random_sample_programs():
    for lang in langs:
        rand_key = random.randint(0, len(programs_dict[lang]))
        code1 = programs_dict[lang][rand_key]
        functions, program_pieces = separate_functions_from_programs(code1, lang)
        func_names = extract_function_names(code1, lang)
        program_piece = " ".join(program_pieces)
        func_call_names = extract_function_calls(program_piece, lang)

        print(lang)
        print(program_piece)
        print("*"*20)
        for func_call in func_call_names:
            if func_call not in sys_calls_dict[lang]:
                print(func_call)
        print("*"*20)
    return
