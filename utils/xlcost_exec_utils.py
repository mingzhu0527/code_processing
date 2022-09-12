from leetcode_exec_utils import *

def generate_java_callcode_no_output_xlcost(codestring, test_case, target_info):
    return_type, target_func_name, para_list = target_info
    codestring = codestring.strip() + "\n"
    codestring += "\tpublic static void main(String args[]){\n"
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
        codestring += target_func_name + "(" + arg_str + ");\n"
        codestring += "\n}\n}"
    else:
        codestring += return_type + " execution_output = " + target_func_name + "(" + arg_str + ");\n"
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
                codestring += "System.out.println(" + execution_output + ");\n"
        else:
            codestring += "System.out.println(execution_output);\n"
        codestring += "\n}\n}"
    return codestring

def generate_cpp_callcode_no_output_xlcost(codestring, test_case, target_info):
    return_type, target_func_name, para_list = target_info
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
    para_str = ",".join([x[0].replace("[]", '') for x in para_list])
    for (arg_name, arg_val), (para_name, para_type) in zip(input_list, para_list):
        arg_str = arg_val.replace('[', '{').replace(']', '}')
        input_str += para_type.replace('ListNode*', "vector<int>").replace('&', "") + " " + para_name + ' = ' + arg_str + ';\n'
    codestring += input_str
    output = target_func_name + "(" + para_str + ")"
    codestring += return_type.replace('ListNode*', "vector<int>") + " execution_output = " + output + ';\n'
    if "vector" in return_type:
        codestring += print_func_name + "(execution_output);\n"
    else:
        codestring += "cout << execution_output << endl;\n"
    codestring += "return 0;\n"
    codestring += "\n}"
    return codestring

def generate_c_callcode_no_output_xlcost(codestring, test_case, target_info):
    return_type, target_func_name, para_list = target_info
    real_return_type, dim = get_real_type_c(return_type)
    print_token = 'd'
    if real_return_type in ['string', 'char']:
        print_token = 's'
    print_func_name = 'print'
    if dim == 1:
        print_str_part1 = "\nvoid print("
        print_str_part2 = """ &vec) {
    printf("[");
    for (int i = 0; i < vec.length; i++){
        if (i < vec.length - 1){
            printf("%PRINT_TOKEN, ", vec[i]);
        } else {
            printf("%PRINT_TOKEN", vec[i])
        }
    }
    printf("]");
}
"""
        print_str = print_str_part1 + return_type + print_str_part2
        print_str = print_str.replace("PRINT_TOKEN", print_token)
        codestring += print_str
    if dim == 2:
        print_func_name = 'print_2d'
        print_str_part1 = "\nvoid print_2d("
        print_str_part2 = """ &vec) {
    printf("[");
    for (int i = 0; i < vec.length; i++){
        printf("[");
        for (int j = 0; j < vec[i].length; j++)
        {
            if (i < vec[i].length - 1){
                printf("%PRINT_TOKEN, ", vec[i][j]);
            } else {
                printf("%PRINT_TOKEN", vec[i][j)
            }
        }
        printf("]");
        if (i < vec.length - 1){
            printf(",");
        }

    }
    printf("]");
}
        """
        print_str = print_str_part1 + return_type + print_str_part2
        print_str = print_str.replace("PRINT_TOKEN", print_token)
        codestring += print_str
    codestring += "\nint main(){}"
    codestring = codestring.strip()[:-1] + '\n'
    test_case = cpp_formatting(test_case)
    input_list = input_parsing(test_case['inputs'])
    if len(input_list) != len(para_list):
        return None
    input_str = ""
    para_str = ",".join([x[0].replace("[]", '') for x in para_list])
    for (arg_name, arg_val), (para_name, para_type) in zip(input_list, para_list):
        arg_str = arg_val.replace('[', '{').replace(']', '}')
        input_str += para_type.replace('ListNode*', "vector<int>").replace('&', "") + " " + para_name + ' = ' + arg_str + ';\n'
    codestring += input_str
    output = target_func_name + "(" + para_str + ")"
    if return_type == "void":
        return None
    codestring += return_type.replace('ListNode*', "vector<int>") + " execution_output = " + output + ';\n'
    if dim > 0:
        codestring += print_func_name + "(execution_output);\n"
    else:
        print_str = """printf("%PRINT_TOKEN\\n", execution_output);\n"""
        print_str = print_str.replace("PRINT_TOKEN", print_token)
        codestring += print_str
    codestring += "return 0;\n"
    codestring += "\n}"
    return codestring

def generate_callcode_no_output_xlcost(program, case, lang, target_info):
    if target_info == None:
        return None
    if lang == 'Python':
        return generate_python_callcode(program, case, target_info)
    elif lang == 'Java':
        return generate_java_callcode_no_output_xlcost(program, case, target_info)
    elif lang == 'C++':
        return generate_cpp_callcode_no_output_xlcost(program, case, target_info)
    elif lang == 'C':
        return generate_c_callcode_no_output_xlcost(program, case, target_info)
    return None

def get_call_dict_no_output_xlcost(code_dic, lang, num_tests=5):
    pieces = code_dic['program_pieces']
    program_prefix = "".join(pieces[:pieces.index(target_function_place_holder)])
    functions = code_dic['functions']
    function_str = "\n".join(functions)
    program = program_prefix + function_str
    target_func_name = code_dic['target_call']
    return_type = code_dic['target_call_return_type']
    target_call_params = code_dic['target_call_params']
    para_list = get_para_list(target_call_params)
    if lang == "C++":
        para_list_testcase = get_para_list_cpp_testcase(target_call_params)
    else:
        para_list_testcase = para_list
    cases = generate_tests(para_list_testcase, num_tests, lang)
    target_info = return_type, target_func_name, para_list
    codestrings = []
    good_cases = []
    for j, case in enumerate(cases):
        codestring = generate_callcode_no_output_xlcost(program, case, lang, target_info)
        if codestring == None:
            continue
        codestrings.append(codestring)
        good_cases.append(case)
    return codestrings, good_cases

def prepare_exec_program_test_generation_xlcost(pids, code_id_lang_dict, lang, num_tests):
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
            codestring_list, good_cases = get_call_dict_no_output_xlcost(code_dic, lang, num_tests=5)
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
    print(len(exception_list))
    return programs, filtered_pids, exception_list, program_id_dict, testcase_id_dict

def generate_testcases_with_diversity_xlcost(leetcode_pids_dict, filtered_code_id_lang_dict, 
                                             num_tests=10, rounds=5):
    testcase_id_lang_dict = {}
    for lang in ['C++', 'Java', "C"]:
        exist_testcase_id_dict = {}
        pids_need_more_test = leetcode_pids_dict[lang]
        print('start', len(pids_need_more_test))
        for i in range(rounds):
            # filtered_pids are pids that successfully generated callcode
            programs, filtered_pids, exception_list, program_id_dict, testcase_id_dict \
                = prepare_exec_program_test_generation_xlcost(pids_need_more_test, 
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
            if len(exist_testcase_id_dict) > 0:
                print("avg num tests", get_average_num_tests(exist_testcase_id_dict))
            print('step', i, len(pids_need_more_test), len(exist_testcase_id_dict))
        testcase_id_lang_dict[lang] = exist_testcase_id_dict
    return testcase_id_lang_dict

