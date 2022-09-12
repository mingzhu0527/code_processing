from leetcode_process_utils import *
from test_generation_utils import *

def get_call_dict_no_output(code_dic, lang, num_tests):
    codestring_list = []
    bad_testcase_list = []
    program = code_dic['program_formatted']
    pid = code_dic['idx']
    target_info = get_target_info(code_dic, lang)
    new_para_list =  target_info[-1]
    cases = generate_tests(new_para_list, num_tests, lang)
    good_cases = []
    for j, case in enumerate(cases):
        codestring = None
        if lang == 'Java':
            codestring = generate_java_callcode_no_output(program, case, target_info)
        elif lang == 'C++':
            codestring = generate_cpp_callcode_no_output(program, case, target_info)
        if codestring == None:
            bad_testcase_list.append((pid, j))
            continue
        codestring_list.append(codestring)
        good_cases.append(case)
    return codestring_list, good_cases

def prepare_exec_program_test_generation(pids, code_id_lang_dict, lang, num_tests):
    program_id_dict = {}
    programs = []
    testcase_id_dict = {}
    no_code_list = []
    exception_list = []
    filtered_pids = []
    idx = 0
    for pid in pids:
        code_dic = code_id_lang_dict[lang][pid]
        try:
            codestring_list, good_cases = get_call_dict_no_output(code_dic, lang, num_tests)
        except:
            exception_list.append(pid)
            continue
        if len(codestring_list) == 0:
            no_code_list.append(pid)
        filtered_pids.append(pid)
        programs += codestring_list
        program_id_dict[pid] = [i for i in range(idx, idx + len(codestring_list))]
        testcase_id_dict[pid] = good_cases
        idx += len(codestring_list)
    return programs, filtered_pids, exception_list, program_id_dict, testcase_id_dict

def filter_testcase_by_output(testcase_id_dict, result_id_dict, result_key_dict, 
                              exist_testcase_id_dict, thres=5):
    pids_need_more_test = []
    existing_outputs = set()
    for pid in result_id_dict.keys():
        result_keys = result_key_dict[pid]
        pid_results = result_id_dict[pid]
        pass_results = []
        for result_key, pid_result in zip(result_keys, pid_results):
            if result_key == 'good':
                # check too large; check empty
                if len(pid_result) > 1000 or len(pid_result.strip()) == 0:
                    continue
                check_empty_list = pid_result.replace('[', 
                                                      '').replace(']', '').replace('(', 
                                                      '').replace(')', '').replace(',', '').strip()
                if len(check_empty_list) == 0:
                    continue
                pass_results.append(pid_result)
        if len(pass_results) == 0:
            pids_need_more_test.append(pid)
            continue
        
        if pid in exist_testcase_id_dict:
            existing_testcases = exist_testcase_id_dict[pid]
            existing_outputs = set([x['output'] for x in existing_testcases])
        testcases = testcase_id_dict[pid]
        
        result_val_set = set()
        filtered_testcases = []
        for result_val, testcase in zip(pass_results, testcases):
            if result_val in result_val_set:
                continue
            if result_val in existing_outputs:
                continue
            testcase['output'] = result_val
            result_val_set.add(result_val)
            filtered_testcases.append(testcase)
        if len(filtered_testcases) > 0:
            if pid in exist_testcase_id_dict:
                exist_testcase_id_dict[pid] += filtered_testcases
            else:
                exist_testcase_id_dict[pid] = filtered_testcases
        # TODO. what about binary outputs?
            if len(exist_testcase_id_dict[pid]) <= 1:
                pids_need_more_test.append(pid)
    return exist_testcase_id_dict, pids_need_more_test

def get_average_num_tests(exist_testcase_id_dict):
    len_list = []
    for pid, tc_list in exist_testcase_id_dict.items():
        len_list.append(len(tc_list))
    return get_avg(len_list)

def generate_testcases_with_diversity(leetcode_pids_dict, filtered_code_id_lang_dict, num_tests=10, rounds=5):
    testcase_id_lang_dict = {}
    for lang in ['C++', 'Java']:
        exist_testcase_id_dict = {}
        pids_need_more_test = leetcode_pids_dict[lang]
        print('start', len(pids_need_more_test))
        for i in range(rounds):
            # filtered_pids are pids that successfully generated callcode
            programs, filtered_pids, exception_list, program_id_dict, testcase_id_dict \
                = prepare_exec_program_test_generation(pids_need_more_test, 
                                                       filtered_code_id_lang_dict, lang, num_tests)
            print("filtered_pids", len(filtered_pids))
            lang_results = p_map(file_executors[lang], programs)
            _ = show_result_summary({lang:lang_results})
            result_id_dict, result_key_dict, error_type_dict = result_mapping(lang_results, program_id_dict, 
                                                                              filtered_pids, lang)
            exist_testcase_id_dict, pids_need_more_test = filter_testcase_by_output(testcase_id_dict, 
                                                                                    result_id_dict, 
                                                                                    result_key_dict, 
                                                                                    exist_testcase_id_dict, 
                                                                                    thres=num_tests)
            print("avg num tests", get_average_num_tests(exist_testcase_id_dict))
            print('step', i, len(pids_need_more_test), len(exist_testcase_id_dict))
        testcase_id_lang_dict[lang] = exist_testcase_id_dict
    return testcase_id_lang_dict


def get_pass_java_types(results_processed, code_id_lang_dict):
    pass_type_dict = {}
    lang = 'Java'
    for i, pid in enumerate(results_processed[lang][0].keys()):
        target_info = get_target_info_java(code_id_lang_dict[lang][pid])
        new_para_list =  target_info[-1]
        return_type = target_info[1]
        for _, pt in new_para_list:
            if pt in pass_type_dict:
                pass_type_dict[pt] += 1
            else:
                pass_type_dict[pt] = 1
        if return_type in pass_type_dict:
            pass_type_dict[return_type] += 1
        else:
            pass_type_dict[return_type] = 1
    return pass_type_dict

def get_all_acceptable_types():
    list_types = ['Boolean', 'Double', 'Integer', 'String', 'Float', 'Long']
    array_types = ['boolean', 'double', 'float', 'String', 'char', 'long', 'int', ]
    node_types = ['ListNode', 'TreeNode']
    dims = [0, 1, 2]
    list_type_list = []
    array_type_list = []
    for tp in list_types:
        list_type_list.append("List<" + tp + ">")
        list_type_list.append("List<List<" + tp + ">>")
    for tp in array_types:
        suf = ""
        for dim in dims:
            suf = "[]" *dim
            array_type_list.append(tp + suf)
    all_types = list_type_list + array_type_list + node_types
    return all_types

def get_leetcode_split(pass_pids_dict):
    leetcode_split_path = cached_path + 'leetcode_split_dict.json'
    if os.path.exists(leetcode_split_path):
        with open(leetcode_split_path) as infile:
            return json.load(infile)
    java_id_set = set(pass_pids_dict['Java'])
    py_id_set = set(pass_pids_dict['Python'])
    cpp_id_set = set(pass_pids_dict['C++'])
    common_ids = list(java_id_set & cpp_id_set & py_id_set)
    import numpy as np
    np.random.shuffle(common_ids)
    train_ratio, val_ratio, test_ratio = 0.7, 0.1, 0.2
    train_num = int(len(common_ids) * train_ratio)
    test_num = int(len(common_ids) * test_ratio)
    val_num = len(common_ids) - train_num - test_num
    train_split = common_ids[:train_num]
    val_split =  common_ids[train_num:train_num+val_num]
    test_split = common_ids[-test_num:]
    assert(len(set(train_split) & set(val_split)) == 0)
    assert(len(set(train_split) & set(test_split)) == 0)
    assert(len(set(val_split) & set(test_split)) == 0)
    assert(len(set(train_split) | set(val_split) | set(test_split)) == len(common_ids))
    leetcode_split_dict = {"train":train_split, "val":val_split, "test":test_split}
    return leetcode_split_dict

def get_pass_pids_dict(call_dict, results_processed):
    pass_pids_dict = {}
    for lang in results_processed.keys():
        programs, program_id_dict, pids = call_dict[lang]
        pass_result_id_dict, result_id_dict, pass_list, fail_list, error_type_dict = results_processed[lang]
        pass_pids = []
        passcount = 0
        pass_symbol = 'True'
        for key, res_list in pass_result_id_dict.items():
            new_res_list = res_list
            if lang == "C++":
                pass_symbol = '1'
            elif lang == "Java":
                new_res_list = [x.strip().split('\n')[-1] for x in res_list]
                pass_symbol = 'true'
            elif lang == "Python":
                new_res_list = [x.strip().split('\n')[-1] for x in res_list]
                pass_symbol = 'True'
            if len(new_res_list) > 0 and new_res_list == [pass_symbol]*len(new_res_list):
                pass_pids.append(key)
        pass_pids_dict[lang] = pass_pids
        print(lang, len(pass_pids), len(pass_pids)/len(pass_list), len(pass_pids)/len(pids))
    return pass_pids_dict

def get_filtered_code_id_lang_dict(pass_pids_dict, code_id_lang_dict):
    filtered_code_id_lang_dict = {}
    for lang in tri_langs:
        lang_dict = code_id_lang_dict[lang]
        filtered_code_id_lang_dict[lang] = {}
        for pid in pass_pids_dict[lang]:
            filtered_code_id_lang_dict[lang][pid] = code_id_lang_dict[lang][pid]
    return filtered_code_id_lang_dict


def py_formatting(test_case):
    for k, v in test_case.items():
        test_case[k] = v.replace("null", "None").replace("true", "True").replace("false", "False")
    return test_case

def java_formatting(test_case):
    for k, v in test_case.items():
        test_case[k] = v.replace("None", "null").replace("True", "true").replace("False", "false")
    return test_case

def cpp_formatting(test_case):
    for k, v in test_case.items():
        test_case[k] = v.replace("nullptr", "null").replace("null", "nullptr")
    return test_case

def prepare_exec_liuyubo_cpp_originial(code_id_lang_dict_liuyubo):
    lang = 'C++'
    programs = []
    program_id_dict = {}
    no_testcase = 0
    pids = list(code_id_lang_dict_liuyubo[lang].keys())
    for i, (pid, code_dic) in enumerate(code_id_lang_dict_liuyubo[lang].items()):
        cpp_imports_list = ['algorithm', 'climits', 'cassert', 'cmath', 'cstring', 'vector']
        imports = "\n".join(["#include <" + x + '>' for x in cpp_imports_list]) + '\n'
        programs.append(imports + code_dic['program_formatted'])
        if len(code_dic['test_cases']['tests']) == 0:
            no_testcase += 1
        program_id_dict[pid] = [i]
    return programs, program_id_dict, pids

def get_target_info_cpp(code_dic):
    code_str = code_dic['program_formatted']
    target_cls_name = get_class_name_java(code_str)
    code_lines = code_str.split("\n")
    sol_class_idx = -1
    public_field_idx = -1
    for i, line in enumerate(code_lines):
        if target_cls_name in line:
            sol_class_idx = i
            break
    for i, line in enumerate(code_lines[sol_class_idx:]):
        if "public:" in line:
            public_field_idx = sol_class_idx + i
            break
    
    func_name_list = code_dic['function_names']
    para_lists = code_dic['parameter_lists']
    return_type_list = code_dic['return_types']
    if len(func_name_list) == 0:
        return None
    if "main" not in func_name_list and len(func_name_list) < len(para_lists):
        func_name_list = ['main'] + func_name_list
    if len(func_name_list) != len(para_lists):
        return None

    bad_funcs = set(["main", "TreeNode", "ListNode", "TreeLinkNode", "Node", "Solution"])
    func_line_idx = -1
    func_idx = -1
    func_dict = {}
    while func_idx + 1 < len(func_name_list):
        func_idx += 1
        target_func_name = func_name_list[func_idx]
        if target_func_name in bad_funcs:
            continue
        para_list = para_lists[func_idx].strip()
        for i, line in enumerate(code_lines[public_field_idx:]):
            if target_func_name in line and para_list in line:
                func_line_idx = public_field_idx + i
                func_dict[func_line_idx] = func_idx
    if len(func_dict) == 0:
        return None
    sorted_index = sorted(list(func_dict.keys()))
    target_func_line_idx = sorted_index[0]
    target_func_idx = func_dict[target_func_line_idx]
    
    target_func_name = func_name_list[target_func_idx]
    return_type = return_type_list[target_func_idx]
    if return_type == "":
        return_type = code_lines[target_func_line_idx].strip().split(' ')[0]
    new_para_list = get_para_list(para_lists[target_func_idx])
    if new_para_list == None:
        return None
    target_info = (target_cls_name, return_type, target_func_name, new_para_list)
    return target_info

def generate_cpp_callcode_no_output(codestring, test_case, target_info):
    target_cls_name, return_type, target_func_name, para_list = target_info
    cpp_imports_list = ['algorithm', 'climits', 'cassert', 'cmath', 'cstring', 'vector']
    imports = "\n".join(["#include <" + x + '>' for x in cpp_imports_list])
    codestring =  imports + '\n' + codestring
    codestring = codestring.replace('int main(', 'int main1(')
    print_func_name = 'print'
    if "vector" in return_type:
        real_type, dim = get_real_type_cpp(return_type)
        if dim == 1:
            print_str_part1 = "\nvoid print("
            print_str_part2 = """ &vec) {
        cout << '[';
        for (int i = 0; i < vec.size(); i++){
            if (i < vec.size() - 1){
                cout << vec[i] << ", ";
            } else {
                cout << vec[i] ;
            }

        }
        cout << ']';
    }
    """
            print_str = print_str_part1 + return_type + print_str_part2
            codestring += print_str
        if dim == 2:
            print_func_name = 'print_2d'
            print_str_part1 = "\nvoid print_2d("
            print_str_part2 = """ &vec) {
        cout << '[';
        for (int i = 0; i < vec.size(); i++){
            cout << '[';
            for (int j = 0; j < vec[i].size(); j++)
            {
                if (j == vec[i].size() - 1) {
                    cout << vec[i][j];
                } else {
                    cout << vec[i][j] << ", ";
                }
            }
            cout << ']';
            if (i < vec.size() - 1){
                cout << ',';
            }

        }
        cout << ']';
    }
            """
            print_str = print_str_part1 + return_type + print_str_part2
            codestring += print_str
    codestring += "\nint main(){}"
    codestring = codestring.strip()[:-1] + '\n'
    test_case = cpp_formatting(test_case)
    input_list = input_parsing(test_case['inputs'])
    if len(input_list) != len(para_list):
        return None
    input_str = ""
    para_str = ",".join([x[0] for x in para_list])
    for (arg_name, arg_val), (para_name, para_type) in zip(input_list, para_list):
        arg_str = arg_val.replace('[', '{').replace(']', '}')
        input_str += para_type.replace('ListNode*', "vector<int>").replace('&', "") + " " + para_name + ' = ' + arg_str + ';\n'
    codestring += input_str
    output = target_cls_name + "()." + target_func_name + "(" + para_str + ")"
    codestring += return_type.replace('ListNode*', "vector<int>") + " execution_output = " + output + ';\n'
    if "vector" in return_type:
        codestring += print_func_name + "(execution_output);\n"
    else:
        codestring += "cout << execution_output << endl;\n"
    codestring += "return 0;\n"
    codestring += "\n}"
    
    return codestring

def generate_cpp_callcode(codestring, test_case, target_info):
    target_cls_name, return_type, target_func_name, para_list = target_info
    cpp_imports_list = ['algorithm', 'climits', 'cassert', 'cmath', 'cstring', 'vector']
    imports = "\n".join(["#include <" + x + '>' for x in cpp_imports_list])
    codestring =  imports + '\n' + codestring
    codestring = codestring.replace('int main(', 'int main1(')
    codestring += "\nint main(){}"
    codestring = codestring.strip()[:-1] + '\n'
    test_case = cpp_formatting(test_case)
    input_list = input_parsing(test_case['inputs'])
    if len(input_list) != len(para_list):
        return None
    input_str = ""
    para_str = ",".join([x[0] for x in para_list])
    for (arg_name, arg_val), (para_name, para_type) in zip(input_list, para_list):
        arg_str = arg_val.replace('[', '{').replace(']', '}')
        input_str += para_type.replace('ListNode*', "vector<int>").replace('&', "") + " " + para_name + ' = ' + arg_str + ';\n'
        
    output_str = ""
    if return_type != "void":
        output_list = [("expected", test_case['output'])]
        return_list = [("expected", return_type)]
        for (arg_name, arg_val), (para_name, para_type) in zip(output_list, return_list):
            arg_str = arg_val.replace('[', '{').replace(']', '}')
            output_str += para_type.replace('ListNode*', "vector<int>").replace('&', "") + " " + para_name + ' = ' + arg_str + ';\n'

    codestring += input_str
    codestring += output_str
    output = target_cls_name + "()." + target_func_name + "(" + para_str + ")"
    codestring += return_type.replace('ListNode*', "vector<int>") + " execution_output = " + output + ';\n'
#     if "vector" in return_type:
#         codestring += "print(execution_output);\n"
#     else:
#         codestring += "cout << execution_output << endl;\n"
    if  return_type != "void":
        codestring += "if (expected == execution_output)\n"
        codestring += "\tcout << true << endl;\n"
        codestring += "else\n"
        codestring += "\tcout << false << endl;\n"
    codestring += "return 0;\n"
    codestring += "\n}"
    
    return codestring

def run_exec_java_test(codestring, timeout=5):
    fn_name = tmp_path + "test" + '.java'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()
    cmd = "java " + fn_name
    output = run_command(cmd, timeout)
    return output

def remove_warning_java(result):
    b = re.sub(r'Note: .*\n', '', result)
    return b

def get_class_name_java(code_str):
    code_lines = code_str.split("\n")
    for i, line in enumerate(code_lines):
        if "Solution" in line:
            line_parts = line.strip().split()
            return line_parts[-2]
    return "Solution"

def get_target_info_java(code_dic):
    code_str = code_dic['program_formatted']
    target_cls_name = get_class_name_java(code_str)
    code_lines = code_str.split("\n")
    sol_class_idx = -1
    public_field_idx = -1
    for i, line in enumerate(code_lines):
        if target_cls_name in line:
            sol_class_idx = i
            break
    
    func_name_list = code_dic['function_names']
    para_lists = code_dic['parameter_lists']
    return_type_list = code_dic['return_types']
    
    if len(func_name_list) == 0:
        return None
    if "main" not in func_name_list and len(func_name_list) < len(para_lists):
        func_name_list = ['main'] + func_name_list
    if len(func_name_list) != len(para_lists):
        return None
    
    bad_funcs = set(["main", "TreeNode", "ListNode", "TreeLinkNode", "__init__", "Solution"])
    func_line_idx = -1
    func_idx = -1
    func_dict = {}
    while func_idx + 1 < len(func_name_list):
        func_idx += 1
        target_func_name = func_name_list[func_idx]
        if target_func_name in bad_funcs:
            continue
        para_list = para_lists[func_idx].strip()
        for i, line in enumerate(code_lines[sol_class_idx:]):
            if target_func_name in line and para_list in line:
                func_line_idx = sol_class_idx + i
                func_dict[func_line_idx] = func_idx
    if len(func_dict) == 0:
        return None
#     print(func_dict)
    sorted_index = sorted(list(func_dict.keys()))
    target_func_line_idx = sorted_index[0]
    target_func_idx = func_dict[target_func_line_idx]
  
    target_func_name = func_name_list[target_func_idx]
    return_type = return_type_list[target_func_idx]
    new_para_list = get_para_list(para_lists[target_func_idx])
    if new_para_list == None:
        return None
    target_info = (target_cls_name, return_type, target_func_name, new_para_list)
    return target_info

# good for testcase generation, but not for callcode
def get_para_list_cpp_testcase(params):
    para_list = params[1:-1].strip()
    new_para_list = []
    if len(para_list) != 0:
        paras = para_list.split(',')
        for para in paras:
            try:
                para_type, para_name = para.strip().rsplit(' ', 1)
                if '[' in para_name and ']' in para_name:
                    dim_left = para_name.count('[')
                    dim_right = para_name.count(']')
                    if dim_left == dim_right:
                        dim = dim_left
                        para_name = para_name.split('[')[0]
                        para_type = para_type + "[]"*dim
                    else:
                        return None
            except:
                return None
            new_para_list.append((para_name.strip(), para_type.strip()))
    return new_para_list

def get_para_list(params):
    para_list = params[1:-1].strip()
    new_para_list = []
    if len(para_list) != 0:
        paras = para_list.split(',')
        for para in paras:
            try:
                para_type, para_name = para.strip().rsplit(' ', 1)
            except:
                return None
            new_para_list.append((para_name.strip(), para_type.strip()))
    return new_para_list

def input_parsing(inputs):
    input_list = []
    lines = inputs.strip().split('\n')
    for line in lines:
        if "=" not in line:
            input_list.append(("unknown", line.strip()))
            continue
        name, val = line.strip().split('=', 1)
        input_list.append((name.strip(), val.strip()))
    return input_list

def output_construction_java(output, return_type):
    if return_type == "void":
        return "", [], None
    output_list = [("expected", output)]
    return_list = [("expected", return_type)]
    return input_construction_java(output_list, return_list)

def get_list_dim(arg_val_eval):
    dim = 1
    if type(arg_val_eval) == list:
        new_l = arg_val_eval
        while len(new_l) > 0 and type(new_l[0]) == list:
            dim += 1
            new_l = new_l[0]
            if type(new_l) != list:
                break
    return dim

def get_arg_str_java(dim, arg_val, arg_val_eval, isstring, ischar):
    quote = "\""
    if ischar:
        quote = "'"
    arg_str = ""
    arg_str_list = []
    if dim == 1:
        arg_val = arg_val.replace("'", quote).replace("\"", quote)
        arg_str = arg_val[1:-1]
    elif dim == 2:
        if isstring or ischar:
            for l in arg_val_eval:
                if type(l) == list:
                    arg_str_list.append(",".join([quote + str(x) + quote for x in l]))
                else:
                    arg_str_list.append(quote + l + quote)
        else:
            for l in arg_val_eval:
                if type(l) == list:
                    arg_str_list.append(",".join([str(x) for x in l]))
                else:
                    arg_str_list.append(str(l))
    return arg_str, arg_str_list

def get_listobj_str_java(dim, arg_str, arg_str_list, islistobj, istree):
    arg_array_java = ""
    if islistobj or istree:
        if dim == 1:
            arg_array_java = "Arrays.asList(" + arg_str + ")"
        elif dim == 2:
            arg_array_java = "Arrays.asList(" + ",".join(["Arrays.asList(" + x + ")" 
                                                          for x in arg_str_list]) + ")"
    else:
        if dim == 1:
            arg_array_java = "{" + arg_str + "}"
        elif dim == 2:
            arg_array_java = "{" + ",".join(["{" + x + "}" for x in arg_str_list]) + "}"
    
    arg_array_java_new = arg_array_java.replace("None", "null").replace("True", "true").replace("False", "false")
    return arg_array_java_new

def parse_list_java(arg_val, para_type):
    islistobj, ischar, isstring, istree = input_listtype_java(para_type)
    imports = []
    print_list = 0
    imports.append("import java.util.Arrays;")
    try:
        arg_val_eval = json.loads(arg_val)
    except:
        arg_val_eval = arg_val
    dim = get_list_dim(arg_val_eval)
    
    arg_str, arg_str_list = get_arg_str_java(dim, arg_val, arg_val_eval, isstring, ischar)
    arg_array_java = get_listobj_str_java(dim, arg_str, arg_str_list, islistobj, istree)
    
    if islistobj:
        if dim == 1:
            print_list = 1
        elif dim == 2:
            print_list = 2
    return arg_array_java, imports, print_list

def input_listtype_java(para_type):
    islistobj = False
    ischar = False
    isstring = False
    istree = False
    if "List" in para_type and para_type != "ListNode":
        islistobj = True
    if "char" in para_type:
        ischar = True
    if "String" in para_type:
        isstring = True
    if "TreeNode" in para_type:
        istree = True
    return (islistobj, ischar, isstring, istree)

def generate_java_callcode_no_output(codestring, test_case, target_info):
    target_cls_name, return_type, target_func_name, para_list = target_info
    codestring = codestring.replace("void main(", "void main1(") 
    codestring = codestring.strip()[:-1]
    codestring += "\tpublic static void main(String args[]){"
    codestring = codestring + "\n" + target_cls_name + " sol = new " + target_cls_name + "();\n"
    
    test_case = java_formatting(test_case)
    
    input_list = input_parsing(test_case['inputs'])
    input_str, input_imports, _ = input_construction_java(input_list, para_list)
    if input_str == None:
        return None
    if len(input_imports) > 0:
        for import_line in input_imports:
            codestring = import_line + '\n' + codestring
    codestring += input_str
    output_imports, print_lists = output_construction_java_no_output(return_type)
    if len(output_imports) > 0:
        for output_line in output_imports:
            codestring = output_line + '\n' + codestring
    arg_names = [x[0] for x in para_list]
    arg_str = ", ".join(arg_names)
    if return_type == "void":
        codestring += "sol." + target_func_name + "(" + arg_str + ");\n"
        codestring += "\n}\n}"
    else:
        codestring += return_type + " execution_output = sol." + target_func_name + "(" + arg_str + ");\n"
        if len(print_lists) > 0:
            print_list = print_lists[0]
            if print_list == 3:
                codestring += "System.out.println(execution_output);\n"
            else:
                print_part1 = "Arrays.toString("
                if print_list == 0:
                    print_part2 = ")"
                else:
                    print_part2 = ".toArray())"
                execution_output = print_part1 + "execution_output" + print_part2
#                 expected = print_part1 + "expected" + print_part2
                codestring += "System.out.println(" + execution_output + ");\n"
        else:
            codestring += "System.out.println(execution_output);\n"
        codestring += "\n}\n}"
    return codestring

def output_construction_java_no_output(para_type):
    if para_type == "void":
        return [], []
    imports = []
    print_lists = []
    _, dim = get_real_type_java(para_type)
    if dim > 0:
        imports.append("import java.util.Arrays;")
        print_lists = [dim]
    if "TreeNode" in para_type or "ListNode" in para_type:
        if "TreeNode" in para_type:
            init_func = "TreeUtils.constructBinaryTree"
            tmp_para_name = para_name + '_tree_para'
            imports.append("import com.fishercoder.common.utils.TreeUtils;")
            imports.append("import java.util.List;")
        if "ListNode" in para_type:
            init_func = "LinkedListUtils.contructLinkedList"
            tmp_para_name = para_name + '_list_para'
            imports.append("import com.fishercoder.common.utils.LinkedListUtils;")
        print_lists = [3]
    return list(set(imports)), print_lists

def input_construction_java(input_list, para_list):
    imports = []
    if len(input_list) != len(para_list):
        return None, imports, None
    input_str = ""
    print_lists = []
    for (arg_name, arg_val), (para_name, para_type) in zip(input_list, para_list):
        arg_str = arg_val
        if arg_val.startswith('[') and arg_val.endswith(']'):
            arg_str, new_imports, print_list = parse_list_java(arg_val, para_type)
            print_lists.append(print_list)
            imports += new_imports
        elif "String" in para_type:
            arg_str = arg_val.replace("'", "\"")
        elif "char" in para_type:
            arg_str = arg_val.replace("\"", "'")
        
        if "TreeNode" in para_type or "ListNode" in para_type:
            if "TreeNode" in para_type:
                init_func = "TreeUtils.constructBinaryTree"
                tmp_para_name = para_name + '_tree_para'
                imports.append("import com.fishercoder.common.utils.TreeUtils;")
                imports.append("import java.util.List;")
                input_str += "List<Integer> " + tmp_para_name + ' = ' + arg_str + ';\n'
            if "ListNode" in para_type:
                init_func = "LinkedListUtils.contructLinkedList"
                tmp_para_name = para_name + '_list_para'
                imports.append("import com.fishercoder.common.utils.LinkedListUtils;")
                input_str += "int[] " + tmp_para_name + ' = ' + arg_str + ';\n'
            print_lists[0] = 3
            input_str += para_type + " " + para_name + " = " + init_func + "(" + tmp_para_name + ");\n"
        else:
            input_str += para_type + " " + para_name + ' = ' + arg_str + ';\n'
    return input_str, list(set(imports)), print_lists




def generate_java_callcode(codestring, test_case, target_info):
    target_cls_name, return_type, target_func_name, para_list = target_info
    codestring = codestring.replace("void main(", "void main1(") 
    codestring = codestring.strip()[:-1]
    codestring += "\tpublic static void main(String args[]){"
    cls_obj_name = ""
    if target_cls_name != "":
        codestring = codestring + "\n" + target_cls_name + " sol = new " + target_cls_name + "();\n"
        cls_obj_name = "sol."
    
    test_case = java_formatting(test_case)
    
    input_list = input_parsing(test_case['inputs'])
    input_str, input_imports, _ = input_construction_java(input_list, para_list)
    if input_str == None:
        return None
    if len(input_imports) > 0:
        for import_line in input_imports:
            codestring = import_line + '\n' + codestring
    codestring += input_str
    output_str, output_imports, print_lists = output_construction_java(test_case["output"], return_type)
    if output_str == None:
        return None
    if len(output_imports) > 0:
        for output_line in output_imports:
            codestring = output_line + '\n' + codestring
    codestring += output_str
    arg_names = [x[0] for x in para_list]
    arg_str = ", ".join(arg_names)
    if return_type == "void":
        codestring += "sol." + target_func_name + "(" + arg_str + ");\n"
        codestring += "\n}\n}"
    else:
        codestring += return_type + " execution_output = " + cls_obj_name + target_func_name + "(" + arg_str + ");\n"
        if len(print_lists) > 0:
#             print(print_lists)
            print_list = print_lists[0]
            if print_list == 3:
                codestring += "System.out.println(execution_output);\n"
                codestring += "if (expected.equals(execution_output))\n"
            else:
                print_part1 = "Arrays.toString("
                if print_list == 0:
                    print_part2 = ")"
                else:
                    print_part2 = ".toArray())"
                execution_output = print_part1 + "execution_output" + print_part2
                expected = print_part1 + "expected" + print_part2
                codestring += "System.out.println(" + execution_output + ");\n"
                if print_list == 0:
                    codestring += "if (Arrays.equals(expected, execution_output))\n"
                else:
                    codestring += "if (expected.equals(execution_output))\n"
        else:
            codestring += "System.out.println(execution_output);\n"
            codestring += "if (expected == execution_output)\n"
        codestring += "\tSystem.out.println(true);\n"
        codestring += "else\n"
        codestring += "\tSystem.out.println(false);"
        codestring += "\n}\n}"
    return codestring


def generate_python_callcode(codestring, test_case, target_info):
    solution, function = target_info
    imports = ['collections', 'itertools', 'math', 'random', 'operator']
    import_str = ""
    for imp in imports:
        import_str += "import " + imp + "\n"
    codestring = import_str + "\n" + codestring
    codestring = codestring + "\nsol = " + solution + "()\n"
    test_case = py_formatting(test_case)
    codestring = codestring + test_case["inputs"] + "\n"
    codestring = codestring + "expected = " + test_case["output"] + "\n"
    input_list = input_parsing(test_case['inputs'])
    var_names = [x[0] for x in input_list]
#     var_names = get_var_names(test_case["inputs"])
    
    is_tree = False
    tree_name = "TreeNode"
    tree_str = py_treenode_code
    listnode_str = py_listnode_code
    if (".left" in codestring and '.right' in codestring) or "TreeNode(" in codestring:
        is_tree = True
    elif (".val" in codestring and '.next' in codestring) or "ListNode(" in codestring:
        is_tree = True
        tree_name = "ListNode"
        tree_str = listnode_str

    if is_tree:
        codestring += tree_str
        inputs = test_case["inputs"]
        lines = inputs.split('\n')
        for line in lines:
            if "=" not in line:
                continue
            varname, var = line.split('=')
            var = var.strip()
            if var.startswith('[') and var.endswith(']'):
                codestring += varname + " = " + tree_name + "(" + varname.strip() + ")\n"
    
    var_names = ", ".join(var_names)
    codestring += "execution_output = sol." + function + "(" + var_names + ")\n"
    codestring += "print(execution_output)\n"
    codestring += "print('*'*10)\n"
    codestring += "if expected == execution_output:\n"
    codestring += "\tprint(True)\n"
    codestring += "else:\n"
    codestring += "\tprint(False)"
    return codestring

def get_class_name_python(code_str):
    code_lines = code_str.split("\n")
    for i, line in enumerate(code_lines):
        if "class Solution" in line:
            line_parts = line.strip().split()
            for line_part in line_parts:
                if line_part.startswith("Solution"):
                    return line_part.replace(" ", "").split("(")[0]
    return "Solution"

def get_target_info_python(code_dic):
    code_str = code_dic['program_formatted']
    function_pat = "def (.*)\("
    class_pat = "class (Solution[0-9]?)\(object\):"
    target_class = "Solution"
    target_func = None
    code_lines = code_str.split("\n")
    for i, line in enumerate(code_lines):
        if "class Solution" in line:
            try:
                class_match = re.findall(class_pat, line)
                target_class = class_match[0]
                sample_lines = "\n".join(code_lines[i+1:i+6])
                function_match = re.findall(function_pat, sample_lines)
                target_func = function_match[0]
                break
            except:
                continue   
    if target_func == None:
        return None
    return (target_class, target_func)


def get_target_info(code_dic, lang):
    if lang == 'Python':
        return get_target_info_python(code_dic)
    elif lang == 'Java':
        return get_target_info_java(code_dic)
    elif lang == "C++":
        return get_target_info_cpp(code_dic)
    return None

def generate_call_code(program, case, lang, target_info):
    if target_info == None:
        return None
    if lang == 'Python':
        return generate_python_callcode(program, case, target_info)
    elif lang == 'Java':
        return generate_java_callcode(program, case, target_info)
    elif lang == 'C++':
        return generate_cpp_callcode(program, case, target_info)
    return None

def get_call_dict(code_dic, lang):
    codestring_list = []
    bad_testcase_list = []
    program = code_dic['program_formatted']
    cases = code_dic['test_cases']['tests']
    pid = code_dic['idx']
    target_info = get_target_info(code_dic, lang)
    for j, case in enumerate(cases):
        codestring = generate_call_code(program, case, lang, target_info)
        if codestring == None:
            bad_testcase_list.append((pid, j))
            continue
        codestring_list.append(codestring)
    return codestring_list, bad_testcase_list
    
def prepare_exec_program(code_id_lang_dict, lang):
    bad_testcase = []
    program_id_dict = {}
    programs = []
    no_code_list = []
    exception_list = []
    idx = 0
    for pid, code_dic in code_id_lang_dict[lang].items():
        try:
            codestring_list, bad_testcase_list = get_call_dict(code_dic, lang)
        except:
            exception_list.append(pid)
            continue
        bad_testcase += bad_testcase_list
        if len(codestring_list) == 0:
            no_code_list.append(pid)
        programs += codestring_list
        program_id_dict[pid] = [i for i in range(idx, idx + len(codestring_list))]
        idx += len(codestring_list)
#     print("bad testcase", len(bad_testcase_list))
#     print("fail to generate call code", len(no_code_list))
#     print("exception_list", len(exception_list))
    pids = list(program_id_dict.keys())
    return programs, program_id_dict, pids

def get_pass_fail_error(results, program_id_dict, pids, lang="Python"):
    result_id_dict, result_key_dict, error_type_dict = result_mapping(results, program_id_dict, pids, lang)
    pass_list = []
    fail_list = []
    for pid in pids:
        result_keys = result_key_dict[pid]
        if len(result_keys) > 0 and result_keys == ['good']*len(result_keys):
            pass_list.append(pid)
        else:
            fail_list.append(pid)
    return result_id_dict, pass_list, fail_list, error_type_dict