from common_utils import *

def get_eval_files(lang1, lang2, data_path, tag='test'):
    lang_pair, lang_pair_path = get_lang_pair_path(lang1, lang2, data_path)
    test_file_prefix = lang_pair_path + tag + "-" + lang_pair + "-tok"
    test_src_file = test_file_prefix + file_extensions[lang1]
    test_tgt_file = test_file_prefix + file_extensions[lang2]
    return test_src_file, test_tgt_file

def get_lang_pair_path(lang1, lang2, data_path):
    lang_pair = lang1 + "-" + lang2
    if not os.path.isdir(data_path + lang_pair):
        lang_pair = lang2 + "-" + lang1
    lang_pair_path = data_path + lang_pair + '/'
    return lang_pair, lang_pair_path

def get_target_map_dict(lang1, lang2, data_path, tag='test'):
    _, lang_pair_path = get_lang_pair_path(lang1, lang2, data_path)
    map_file = lang_pair_path + tag + "-map.jsonl"
    map_dict = {}
    with open(map_file) as infile:
        lines = infile.readlines()
    map_dict = {i:x.strip() for i, x in enumerate(lines)}
    reverse_map_dict =  {v: k for k, v in map_dict.items()}
    return map_dict, reverse_map_dict

def checkEncoding(file_path):
    with open(file_path, 'rb') as rawdata:
        result = chardet.detect(rawdata.read(10000))
    return result['encoding']

def fix_encoding(text):
    y = unicodedata.normalize('NFKD',text)
    y = y.encode('ASCII', 'ignore').decode('ascii', 'ignore')
    return y

def remove_markdown(html_str):
#     html = markdown(html_str)
#     text = ''.join(BeautifulSoup(html).findAll(text=True))
    text = BeautifulSoup(html_str, "lxml").text
    text = unicodedata.normalize("NFKD",text)
    return text

def remove_all_comments(string, lang):
    if lang == 'Python':
        pattern = "(?:'''[\s\S]*''')|(?:\"\"\"[\s\S]*\"\"\")|(?:''[\s\S]*''')|(?:#[^\n]*)"
    else:
        pattern = '(?://[^\n]*|/\*(?:(?!\*/).)*\*/)'
    reg = re.compile(pattern, re.DOTALL)
    comments = reg.findall(string)
    newstring = reg.sub("", string)
    return newstring, comments

def format_codestring(codestring, lang):
    lang1_code, _ = remove_all_comments(codestring, lang)
    lang1_code_format = fix_program_format(lang1_code)
    lang1_code_enc = fix_encoding(lang1_code_format)
    codestring_formatted = re.sub('[\n]{2,}','\n', lang1_code_enc)
    return codestring_formatted

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
    
    return string, comment

def fix_program_format(y):
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

def print_src(src):
    lines = src.split('\n')
    for i, line in enumerate(lines):
        print(i+1, line)
    return

def mono_get_all_data(lang1, lang2, data_type):
    json_dict = {}
    for tag in tags:
        json_dict[tag] = sample_code(lang1, lang2, data_type, tag)
    return json_dict

def get_all_data(data_type):
    json_dicts = {}
    for lang1 in langs + ['comment', 'desc']:
        for lang2 in langs:
            if lang1 == lang2:
                continue
            json_dicts[(lang1, lang2)] = mono_get_all_data(lang1, lang2, data_type)
    return json_dicts

def get_all_programs(program_id_lang_dic):
    programs_dict = {}
    pids_dict = {}
    for lang, lang_dic in program_id_lang_dic.items():
        programs = []
        detoked_programs = []
        pids = list(lang_dic.keys())
        for pid in pids:
            v = lang_dic[pid]
            program = v['program_formatted']
            programs.append(program)
        programs_dict[lang] = programs
        pids_dict[lang] = pids
    return programs_dict, pids_dict

def get_all_programs_detok_from_cache(program_id_lang_dic, pids_dict_path, 
                                      programs_dict_path, programs_toked_dict_path, programs_detoked_dict_path):
    if os.path.exists(programs_dict_path) \
        and os.path.exists(programs_detoked_dict_path) \
        and os.path.exists(pids_dict_path) \
        and os.path.exists(programs_toked_dict_path):
        with open(pids_dict_path) as infile:
            pids_dict = json.load(infile)
        with open(programs_dict_path) as infile:
            programs_dict = json.load(infile)
        with open(programs_toked_dict_path) as infile:
            programs_toked_dict = json.load(infile)
        with open(programs_detoked_dict_path) as infile:
            programs_detoked_dict = json.load(infile)
    else:
        pids_dict = {}
        programs_dict = {}
        programs_toked_dict = {}
        programs_detoked_dict = {}
        for lang, lang_dic in program_id_lang_dic.items():
            programs = []
            toked_programs = []
            detoked_programs = []
            pids = list(lang_dic.keys())
            for pid in tqdm(pids):
                v = lang_dic[pid]
                program = v['program_formatted']
                program_tokenized = " ".join(v['tokens'])
                program_detok = detok_format(program_tokenized, file_detokenizers[lang])
                programs.append(program)
                toked_programs.append(program_tokenized)
                detoked_programs.append(program_detok)
            programs_dict[lang] = programs
            programs_toked_dict[lang] = toked_programs
            programs_detoked_dict[lang] = detoked_programs
            pids_dict[lang] = pids
        with open(pids_dict_path, 'w') as outfile:
            json.dump(pids_dict, outfile)
        with open(programs_dict_path, 'w') as outfile:
            json.dump(programs_dict, outfile)
        with open(programs_toked_dict_path, 'w') as outfile:
            json.dump(programs_toked_dict, outfile)
        with open(programs_detoked_dict_path, 'w') as outfile:
            json.dump(programs_detoked_dict, outfile)
    return pids_dict, programs_dict, programs_toked_dict, programs_detoked_dict

def get_all_programs_detok(program_id_lang_dic):
    pids_dict = {}
    programs_dict = {}
    programs_detoked_dict = {}
    for lang, lang_dic in program_id_lang_dic.items():
        programs = []
        detoked_programs = []
        pids = list(lang_dic.keys())
        for pid in tqdm(pids):
            v = lang_dic[pid]
            program = v['program_formatted']
            
            program_tokenized = " ".join(v['tokens'])
            if lang == "Javascript":
                program_tokenized = " ".join(file_tokenizers[lang](program))
            program_detok = detok_format(program_tokenized, file_detokenizers[lang])
            programs.append(program)
            detoked_programs.append(program_detok)
        programs_dict[lang] = programs
        programs_detoked_dict[lang] = detoked_programs
        pids_dict[lang] = pids
    return pids_dict, programs_dict, programs_detoked_dict

def sample_code(lang1, lang2, data_type, tag):
    lang_pair = lang1 + "-" + lang2
    if lang1 == "desc":
        code_path_prefix = mono_program_data
    elif lang1 == "comment":
        code_path_prefix = mono_snippet_data
    elif data_type == "snippet":
        code_path_prefix = snippet_data
    else:
        code_path_prefix = program_data
    if not os.path.exists(code_path_prefix + lang_pair):
        lang_pair = lang2 + "-" + lang1

    fn = tag + "-" + lang_pair + file_extensions[lang2]
    new_fn = data_type + "-" + fn + ".json"
    code_path = code_path_prefix + lang_pair + "/" + fn
    
    if not os.path.exists(code_viewer_path + new_fn):
        format_hypo_new(code_path, code_viewer_path + new_fn, lang2)
    with open(code_viewer_path + new_fn) as infile:
        jsons = json.load(infile)
    return jsons

def get_diff(original, modified):
    for i, (f1, f2) in enumerate(zip(original, modified)):
        show_comparison(f1, f2,  sidebyside=False)
        print('----------------------')
    return

def format_raw_code(hypo_collection_path, exp_prefix):
    fns = os.listdir(hypo_collection_path)
    for fn in fns:
        if fn.startswith(exp_prefix):
            hypo_path = hypo_collection_path + fn + "/"
            generate_json_hypo(hypo_path)
    return

def print_formatted_code(path, is_print=True):
    with open(path) as infile:
        code_fmt = json.load(infile)
        code_fmt_list = []
        for i in range(len(code_fmt)):
            cf = code_fmt[str(i)]
            code_fmt_list.append(cf)
        if is_print:
            for cf in code_fmt_list:
                print(cf)
                print("--------------------")
    return code_fmt_list




def format_hypo_new(hypo_fn, new_fn, lang):
    with open(hypo_fn) as infile:
        lines = infile.readlines()
        hypo_json = {}
        for i, line in enumerate(lines):
            line = line.replace("@@ ", "")
            hypo_json[i] = detok_format(line, file_detokenizers[lang])
    with open(new_fn, "w") as outfile:
        json.dump(hypo_json, outfile)
    return

def format_hypo(hypo_fn, lang):
    with open(hypo_fn) as infile:
        lines = infile.readlines()
        hypo_json = {}
        for i, line in enumerate(lines):
            hypo_json[i] = detok_format(line, file_detokenizers[lang])
    with open(hypo_fn[:-4] + ".json", "w") as outfile:
        json.dump(hypo_json, outfile)
    return
        
def generate_json_hypo(hypo_path):
    hypos = os.listdir(hypo_path)
    for hypo in hypos:
        hypo_fn = hypo_path + hypo
        if "_sa-" in hypo and hypo.endswith("txt"):
    #         hyp0.python_sa-cpp_sa.test_beam0_5_0.9_54.36.txt
#             if not os.path.exists(hypo_fn[:-4] + ".json"):
            lang_low = hypo.split(".")[1].split("-")[1].split("_")[0]
            lang = lang_upper[lang_low]
            format_hypo(hypo_path + hypo, lang)
    return

def read_tok_file():
    id_lang_dic = {}
    program_lang_dic = {}
    json_dict = {}
    for lang in file_tokenizers.keys():
        with open(map_data_path + lang + "-mapping-tok.jsonl") as infile:
            lines = infile.readlines()
            jsons = [json.loads(line.strip()) for line in lines]
            json_dict[lang] = jsons
            id_dic = {}
            program_dic = {}
            for js in jsons:
                idx = js['idx'].replace(lang, '')
                pid_l = js['idx'].split('-')
                pid = pid_l[0]
                cid = int(pid_l[-1])
                id_dic[idx] = js
                if pid not in program_dic:
                    program_dic[pid] = set()
                program_dic[pid].add(cid)
            id_lang_dic[lang] = id_dic
            program_lang_dic[lang] = program_dic
    return json_dict, id_lang_dic, program_lang_dic

def read_tok_file_filtered():
    """
    This one is for stats collection. Thus removed empty lines.
    """
    id_lang_dic = {}
    program_lang_dic = {}
    json_dict = {}
    for lang in file_tokenizers.keys():
        with open(map_data_path + lang + "-mapping-tok.jsonl") as infile:
            lines = infile.readlines()
            jsons = [json.loads(line.strip()) for line in lines]
            json_dict[lang] = []
            id_dic = {}
            program_dic = {}
            for js in jsons:
                if js['bpe'] == '':
                    continue
                idx = js['idx'].replace(lang, '')
                pid_l = js['idx'].split('-')
                pid = pid_l[0]
                cid = int(pid_l[-1])
                id_dic[idx] = js
                json_dict[lang].append(js)
                if pid not in program_dic:
                    program_dic[pid] = set()
                program_dic[pid].add(cid)
            id_lang_dic[lang] = id_dic
            program_lang_dic[lang] = program_dic
    return json_dict, id_lang_dic, program_lang_dic