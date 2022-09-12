import os
from tqdm import tqdm
import pickle
import numpy as np
np.random.seed(0)
import random
import json
import re
import string
from collections import Counter


def generate_ints(int_lower_bound, int_higher_bound, num_tests):
    return list(np.random.randint(int_lower_bound, int_higher_bound+1, num_tests))

def generate_strings(length_upper_bound, num_tests, equal_lengths = False):
    letters = string.ascii_lowercase
    strings = []
    for i in range(num_tests):
        if equal_lengths:
            length = length_upper_bound
        else:
            length = np.random.randint(1, length_upper_bound+1, 1)[0]
        strings.append(''.join(random.choice(letters) for j in range(length)))
    return strings

def generate_chars(num_tests):
    chars = []
    letters = string.ascii_lowercase
    for i in range(num_tests):
        chars.append(random.choice(letters))
    return chars

def generate_floats(float_lower_bound, float_higher_bound, num_tests):
    return np.random.uniform(float_lower_bound, float_higher_bound,[num_tests]).tolist()

def generate_booleans(num_tests):
    booleans = np.random.choice(a=[False, True], size=(num_tests,), p=[0.60, 0.4])
    return booleans


def generate_primitive_list(data_type, lower_bound, uppper_bound, num_cols):
    if data_type in ["int", 'short', 'long']:
        return generate_ints(lower_bound, uppper_bound, num_cols)
    elif data_type in ['float', 'double']:
        return generate_floats(lower_bound, uppper_bound, num_cols)
    elif data_type in ['String', 'string']:
        return generate_strings(10, num_cols)
    elif data_type in ['char']:
        return generate_chars(num_cols)
    elif data_type in ['boolean', 'bool']:
        return generate_booleans(num_cols)
    return []

# int long float double boolean char
def generate_primitive(dim, data_type, num_tests, lower_bound, uppper_bound, max_row=10, max_len=10, num_cols=-1):
    if dim == 1:
        return generate_primitive_list(data_type, lower_bound, uppper_bound, num_tests)
    elif dim == 2:
        result = []
        for _ in range(num_tests):
            num_cols = random.randint(1, max_len)
            result.append(generate_primitive_list(data_type, lower_bound, uppper_bound, num_cols))
        return result
    elif dim == 3:
        result = []
        for _ in range(num_tests):
            rows = []
            num_rows, num_cols = random.randint(1, max_row), random.randint(1, max_len)
            for _ in range(num_rows):
                rows.append(generate_primitive_list(data_type, lower_bound, uppper_bound, num_cols))
            result.append(rows)
        return result
    
def get_real_type_c(input_type):
    if input_type.endswith('&'):
        input_type = input_type[:-1]
    inputs = input_type.split()
    if len(inputs) > 1 and inputs[0] == 'const':
        input_type = ' '.join(inputs[1:])
    if input_type.endswith('*'):
        data_type = input_type.replace(' ', '').replace("*", "")
        dim = input_type.count("*")
    else:
        data_type = input_type.replace(' ', '').replace("[]", '')
        dim = input_type.replace(' ', '').count("[]")
    prim_types = set(['bool', 'double', 'float', 'string', 'String', 'char', 'long', 'int'])
    if data_type in prim_types:
        if data_type == 'char' and dim == 1:
            return 'string', 0
        return data_type, dim
    elif input_type in ["long long", "long long int",'uint32_t', 'unsigned int', 'unsigned', 'int const']:
        data_type = 'int'
    elif input_type in ["string&", 'const string&', "const string"]:
        data_type = 'string'
    elif "vector<" in input_type:
        dim = input_type.count("vector")
        typ = input_type.replace("vector", "").replace("<", "").split('>')[0]
        data_type = typ
    else:
        data_type = None
    
    return data_type, dim

def get_real_type_cpp(input_type):
    if input_type.endswith('&'):
        input_type = input_type[:-1]
    inputs = input_type.split()
    if len(inputs) > 1 and inputs[0] == 'const':
        input_type = ' '.join(inputs[1:])
    if input_type.endswith('*'):
        data_type = input_type.replace(' ', '').replace("*", "")
        dim = input_type.count("*")
    else:
        data_type = input_type.replace(' ', '').replace("[]", '')
        dim = input_type.replace(' ', '').count("[]")
    prim_types = set(['bool', 'double', 'float', 'string', 'String', 'char', 'long', 'int'])
    if data_type in prim_types:
        return data_type, dim
    elif input_type in ["long long", "long long int",'uint32_t', 'unsigned int', 'unsigned', 'int const']:
        data_type = 'int'
    elif input_type in ["string&", 'const string&', "const string"]:
        data_type = 'string'
    elif "vector<" in input_type:
        dim = input_type.count("vector")
        typ = input_type.replace("vector", "").replace("<", "").split('>')[0]
        data_type = typ
    else:
        data_type = None
    return data_type, dim

def get_real_type(input_type, lang):
    if lang == "Java":
        return get_real_type_java(input_type)
    if lang == "C#":
        return get_real_type_csharp(input_type)
    if lang == "C++":
        return get_real_type_cpp(input_type)
    if lang == "C":
        return get_real_type_c(input_type)
    return None, 0

def get_real_type_csharp(input_type):
    data_type = input_type.replace(' ', '').replace("[]", '')
    dim = input_type.replace(' ', '').count("[]")
    prim_types = set(['boolean', 'double', 'float', 'string', 'String', 'char', 'long', 'int'])
    if data_type in prim_types:
        return data_type, dim
    elif input_type in ["TreeNode", 'ListNode']:
        data_type = 'int'
        dim = 1
    elif "List<" in input_type:
        dim = input_type.count("List")
        typ = input_type.replace(">", "").split('<')[-1]
        if typ in ["Integer"]:
            data_type = 'int'
        elif typ in["TreeNode", 'ListNode']:
            data_type = 'int'
            dim += 1
        else:
            data_type = typ.lower()
    elif "IEnumerable<" in input_type:
        dim = input_type.count("IEnumerable")
        typ = input_type.replace(">", "").split('<')[-1]
        if typ in ["Integer"]:
            data_type = 'int'
        else:
            data_type = typ.lower()
    else:
        data_type = None
    return data_type, dim

def get_real_type_java(input_type):
    data_type = input_type.replace(' ', '').replace("[]", '')
    dim = input_type.replace(' ', '').count("[]")
    prim_types = set(['boolean', 'double', 'float', 'string', 'String', 'char', 'long', 'int'])
    if data_type in prim_types:
        return data_type, dim
    elif input_type in ["TreeNode", 'ListNode']:
        data_type = 'int'
        dim = 1
    elif "List<" in input_type:
        dim = input_type.count("List")
        typ = input_type.replace(">", "").split('<')[-1]
        if typ in ["Integer"]:
            data_type = 'int'
        elif typ in["TreeNode", 'ListNode']:
            data_type = 'int'
            dim += 1
        else:
            data_type = typ.lower()
    else:
        data_type = None
    return data_type, dim

def generate_tests(input_list, num_tests, lang):
    real_type_dict = {}
    for var_name, input_type in input_list:
        data_type, dim = get_real_type(input_type, lang)
        if data_type == None:
            return []
        real_type_dict[var_name] = (data_type, dim)
        if not data_type:
            return []
    gen_input_dict = {}
    max_len=10
    test_cases = [{'inputs':""} for i in range(num_tests)]
    for var_name, _ in input_list:
        data_type, dim = real_type_dict[var_name]
        inputs = generate_primitive(1+dim, data_type, num_tests, 0, 100, max_row=10, 
                                                      max_len=max_len, num_cols=-1)
        for i in range(num_tests):
            input_str = var_name + ' = ' + repr(inputs[i]) + '\n'
            input_str = input_str.replace("'", '\"')
            test_cases[i]['inputs'] += input_str
    return test_cases
    
