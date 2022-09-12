#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import os
import re
import jsonlines
import unicodedata
import chardet
from collections import Counter
from functools import partial
import csv



def checkEncoding(file_path):
    with open(file_path, 'rb') as rawdata:
        result = chardet.detect(rawdata.read(10000))
    return result['encoding']

def get_problem_desc(file_path):
    file_encoding = checkEncoding(file_path)
    with open(file_path, 'r',  encoding = file_encoding) as temp_f:
        lines = temp_f.readlines()
        line = lines[0]  
        problem_statement = line.split('-')[0].strip()
#         print(problem_statement)
    return problem_statement

def read_file(data_file):
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        largest_column_count = 0
        for i, row in enumerate(csv_reader):
            if i == 0:
                problem_statement = row[0].split('-')[0].strip()
            column_count = len(row) + 1           
            largest_column_count = column_count if largest_column_count < column_count else largest_column_count

    column_names = [i for i in range(0, largest_column_count)]    
    df = pd.read_csv(data_file, header=None, skiprows=[0],
                     names=column_names)
    return df, problem_statement

def read_file1(data_file):
    data_file_delimiter = ','
    file_encoding = checkEncoding(file_path)
    largest_column_count = 0
    with open(data_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=data_file_delimiter)
        for i, row in enumerate(csv_reader):
            if i == 0:
                problem_statement = row[0].split('-')[0].strip()
            column_count = len(row) + 1     
            largest_column_count = column_count if largest_column_count < column_count else largest_column_count
    
    column_names = [i for i in range(0, largest_column_count)]    
    df = pd.read_csv(data_file, header=None, 
                     skiprows=[0],
                     delimiter=data_file_delimiter, 
                     names=column_names, 
                     encoding = file_encoding)
    return df, problem_statement

def read_file_orig(file_path):
    data_file = os.path.join(file_path)
    data_file_delimiter = ','
    largest_column_count = 0

    file_encoding = checkEncoding(file_path)
    with open(data_file, 'r',  encoding = file_encoding) as temp_f:
        lines = temp_f.readlines()
        for l in lines:            
            column_count = len(l.split(data_file_delimiter)) + 1           
            largest_column_count = column_count if largest_column_count < column_count else largest_column_count
    
    column_names = [i for i in range(0, largest_column_count)]    
    df = pd.read_csv(data_file, header=None, 
                     delimiter=data_file_delimiter, 
                     names=column_names, 
                     encoding = file_encoding)
    return df

def remove_comments(string, lang):
    if lang == 'Python':
        pattern = "(?:'''[\s\S]*''')|(?:''[\s\S]*''')"
    else:
        pattern = "\/\*[\s\S]*\*\/"
    #comment = re.findall(pattern, string)
    search_result = re.search(pattern, string)
    if(isinstance(search_result, type(None))):
        comment = ""
    else:
        comment = search_result.group(0)
    string = re.sub(pattern, '', string)
    #print(comment)
    
    return string, comment

# def normalise_code(text):
#     #y = [unicodedata.normalize('NFKD',i) for i in text]
#     y = unicodedata.normalize('NFKD',text)
#     y = y.encode('ASCII', 'ignore').decode('ascii', 'ignore')
#     y = y.replace('\\xa0', '')
#     z = re.sub('[ ]{2,}|\n','',y)
#     return z

def fix_encoding(text):
    y = unicodedata.normalize('NFKD',text)
    y = y.encode('ASCII', 'ignore').decode('ascii', 'ignore')
#     z = re.sub('[ ]{2,}',' ',y)
    return y

def fix_format_python(y):
    y = re.sub("[ ]{4}",'\t', y)
    y = re.sub('\t[ ]+','\t', y)
    x = re.sub('[\n]{2,}','\n',y)
    xs = x.split('\n')
#     remove empty lines
    s = "\n".join([t for t in xs if len(t.strip()) > 0])
    if "(" in s and ")" in s:
#     remove linebreak and \t between ()
        s = re.sub(r'\n(?=[^()]*\))', '', s)
        s = re.sub(r'[\t ]+(?=[^()]*\))', ' ', s)
    if "[" in s and "]" in s:
#     remove linebreak between []
        s = re.sub(r'\n(?=[^\[\]]*\])', '', s)
        s = re.sub(r'[\t ]+(?=[^\[\]]*\])', ' ', s)
#     reduce unnecessary black space
    s = re.sub('[ ]{2,}',' ',s)
    return s

def fix_format(y):
    y = re.sub("[\xa0]{4}",'\t', y)
    y = re.sub('[\xa0]+',' ', y)
    y = re.sub('\t[ ]+','\t', y)
    x = re.sub('[\n]{2,}','\n',y)
    xs = x.split('\n')
#     remove empty lines
    s = "\n".join([t for t in xs if len(t.strip()) > 0])
#     remove linebreak between ()
    s = re.sub(r'\n(?=[^()]*\))', '', s)
#     remove linebreak between []
    s = re.sub(r'\n(?=[^\[\]]*\])', '', s)
    s = re.sub(r'[\t ]+(?=[^()]*\))', ' ', s)
    s = re.sub(r'[\t ]+(?=[^\[\]]*\])', ' ', s)
#     reduce unnecessary black space
    s = re.sub('[ ]{2,}',' ',s)
    return s

def fix_newline_tab(s):
    s = re.sub('[\t|\n]',' ', s)
    s = re.sub('[ ]{2,}',' ',s)
    return s

def raw(string: str, replace: bool = False) -> str:
    """Returns the raw representation of a string. 
    If replace is true, replace a single backslash's repr \\ with \."""
    r = repr(string)[1:-1]  # Strip the quotes from representation
    if replace:
        r = r.replace('\\\\', '\\')
    return r

def count_tabs(string):
    t_count = 0
    for c in string:
        if(c !='\t'):
            break
        t_count+=1
    return t_count

def remove_python_tabs3(string):
    x = string.strip('\n')
    x = x.split('\n')
    min_indent = count_tabs(x[0])
    indent_l = [count_tabs(y) for y in x]
    x = [snippet[min(num, min_indent):] for snippet, num in zip(x, indent_l)]
    x = '\n'.join(x)
    return x

def remove_python_tabs2(string):
    x = string.strip('\n')
    x = x.split('\n')
    min_indent = min([count_tabs(y) for y in x])
    x = [snippet[min_indent:] for snippet in x]
    x = '\n'.join(x)
    return x

def remove_python_tabs(code):
    code = code.strip('\n')
    t_count = 0
    for c in code:
        if(c !='\t'):
            break
        t_count+=1
    pattern = '[\t]{' + str(t_count)+',}'
    
    def evaluate(match, length):
        if(length == 0):
            return match.group()
        return match.group()[:-length]

    code_out = re.sub(pattern, partial(evaluate, length=t_count), code)
    return code_out


def get_map_data(data_path, output_path):
    data_files = os.listdir(data_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    map_files = []
    program_files = []

    for lang1 in file_extensions.keys():
        lang1_map_file  = jsonlines.open(os.path.join(output_path, lang1 + '-mapping.jsonl'), "w")
        lang1_program_file = jsonlines.open(os.path.join(output_path, lang1 + '-program.jsonl'), "w")
        map_files.append(lang1_map_file)
        program_files.append(lang1_program_file)

    count = 0
    for file in data_files:
        if not file.endswith('.csv'):
            continue
        file_path = os.path.join(data_path, file)    
        df, problem_statement = read_file(file_path)
        df = df.set_index(df.columns[0])

        for lang1, lang1_map_file, lang1_program_file in zip(file_extensions.keys(), map_files, program_files):
            lang1_data = df.loc[lang1]
            program_list = []
            sids = []
            for idx in df.columns:
                if pd.isna(lang1_data.loc[idx]):
                    continue;

                lang1_code, lang1_comment = remove_comments(lang1_data.loc[idx], lang1)
                if lang1_code.strip()== "":
                    continue;

                lang1_comment = fix_encoding(fix_format(lang1_comment))
                lang1_code_format = fix_format_python(lang1_code)
                lang1_code_enc = fix_encoding(lang1_code_format)
                if lang1 == "PHP" and len(program_list) == 0:
                    lang1_code_enc = "<?php\n" + lang1_code_enc
                program_list.append(lang1_code_enc)
                if(lang1 == "Python"):
                    lang1_snippet = remove_python_tabs3(lang1_code_enc)
                else:
                    lang1_snippet = lang1_code_enc.replace("\t", "")

                lang1_key = file.split('.')[0] + '-' + lang1 + '-' + str(idx)
                sids.append(idx)
                lang1_map_file.write(dict(idx = lang1_key, code = lang1_code,
                                          comment = lang1_comment, problem_desc = problem_statement,
                                         snippet = lang1_snippet))
            if len(program_list) > 0:
                program_src = re.sub('[\n]{2,}','\n', "\n".join(program_list))
                program_src = program_src.strip()
                program_src_no_newline = fix_newline_tab(program_src)
                final_program_src = program_src_no_newline
    # We remove newline and indentation for all languages except for Python. Transcoder did the same thing. 
    # For future reference of the original code, we keep the formatted code in the program_formatted
    # To compile, we need to use the detokenizer. The original detokenizer need some improvements
    # Transcoder doesn't have the problem of #define MAX 5, because it only evaluate functions

    # 去掉newline以后，在一些语言的program里导致了tokenization error
    # 决定保留newline（和Indentation）。反正对于其它language，tokenize时会自动被tokenizer去掉
                if lang1 == "Python":
                    final_program_src = program_src
                lang1_program_file.write(dict(idx = file.split('.')[0] + '-' + lang1, 
                                              program = final_program_src,
                                              program_formatted = program_src,
                                              snippet_ids = sids,
                                              problem_desc = problem_statement,
                                             ))
    for lang1_map_file, lang1_program_file in zip(map_files, program_files):
        lang1_map_file.close()
        lang1_program_file.close()
    return

