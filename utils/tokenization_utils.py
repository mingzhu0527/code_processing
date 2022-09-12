# from codegen_sources.model.translate import *
# from codegen_sources.model.preprocess import *
from common_utils import *
from transcoder_utils import *
from program_utils import *
from diff_utils import *
from split_utils import *
from py715_bug_utils import *

def get_length_lang_dict(prepro_program_dict, tokenizer, use_cache=True):
    if use_cache and os.path.exists(cached_path + "xlcost_exec_filtered_pids_dict.pkl"):
        with open(cached_path + "xlcost_exec_filtered_pids_dict.pkl", 'rb') as infile:
            length_lang_dict = pickle.load(infile)
            return length_lang_dict
    length_lang_dict = {lang:[] for lang in langs}
    for lang in prepro_program_dict.keys():
        programs = prepro_program_dict[lang]
        for program in programs:
            input_ids = tokenizer(program)['input_ids']
            length_lang_dict[lang].append(len(input_ids))
        print(lang, max(length_lang_dict[lang]), min(length_lang_dict[lang]), get_avg(length_lang_dict[lang]))
    return length_lang_dict

def notok_prepro_parallel(inputs):
    program, lang, is_plbart = inputs
    if lang != "Python":
        if lang == "Java":
            new_program = " ".join(program.split())
            if is_plbart:
                new_program = new_program.replace('java', 'J_TOKEN')
        else:
            new_program = " ".join(program.replace('\n', ' NEW_LINE ').split())
    else:
        try:
            new_program = " ".join(file_tokenizers[lang](program))
        except:
            new_program = ""
    return new_program

def notok_prepro(program, lang, is_plbart=False):
    if lang != "Python":
        if lang == "Java":
            new_program = " ".join(program.split())
            if is_plbart:
                new_program = new_program.replace('java', 'J_TOKEN')
        else:
            new_program = " ".join(program.replace('\n', ' NEW_LINE ').split())
    else:
        try:
            new_program = " ".join(file_tokenizers[lang](program))
        except:
            new_program = ""
    return new_program

def data_prepro_notok(programs_dict, programs_toked_dict, is_plbart=False):
    new_program_dict = {}
    for lang in programs_dict.keys():
        programs = programs_dict[lang]
        new_programs = []
        if lang != "Python":
            for program in programs:
                new_program = notok_prepro(program, lang, is_plbart)
                new_programs.append(new_program)
        else:
            new_programs = programs_toked_dict[lang]
        new_program_dict[lang] = new_programs  
    return new_program_dict

def notok_detok_parallel(inputs):
    codestring, lang, is_plbart = inputs
    if lang == 'Python':
        codestring = detok_format(codestring, file_detokenizers[lang])
    else:
        codestring = codestring.replace('NEW_LINE', '\n')
    if is_plbart:
        if lang == "Java":
            codestring = codestring.replace('J_TOKEN', 'java')
    return codestring

def notok_detok(codestring, lang, is_plbart=False):
    if lang == 'Python':
        codestring = detok_format(codestring, file_detokenizers[lang])
    else:
        codestring = codestring.replace('NEW_LINE', '\n')
    if is_plbart:
        if lang == "Java":
            codestring = codestring.replace('J_TOKEN', 'java')
    return codestring

def fix_format(y):
    y = y.replace(" @ @", "@@").replace("@ @", "@@")
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
#     s = re.sub('[ ]{2,}',' ',s)
    return s

def detok_format(x, detokenizer):
    detoc_seq = detokenizer(x)
    fixed_seq = fix_format(detoc_seq)
    return fixed_seq

def tokenizer_exec_check(tokenizer, new_program_dict, model_type='codet5', num_dp=-1):
    decode_program_dict = {}
    for lang in new_program_dict.keys():
        programs = new_program_dict[lang]
        if num_dp > -1:
            programs = programs[:num_dp]
        new_programs = []
        for program in tqdm(programs):
            is_plbart = False
            if model_type == 'plbart':
                if lang == "Java":
                    is_plbart = True
                    program = program.replace('java', 'J_TOKEN')
            input_ids = tokenizer(program)['input_ids']
            new_program = tokenizer.decode(input_ids, skip_special_tokens=True)
            new_program = notok_detok(new_program, lang, is_plbart=is_plbart)
            new_programs.append(new_program)
        decode_program_dict[lang] = new_programs
    return decode_program_dict



# 除了Python, 应该所有的indentation都可以去掉
# Java需要newline换成空格
# C++ 和C 需要include后面的newline
# JS 需要newline
# PHP完全不能tokenize
def get_pair_data_notok(pair_path, dataset_name, pids_dict, new_program_dict, test_list, val_list):
    pair_path += dataset_name + '/'
    if not os.path.exists(pair_path):
        os.makedirs(pair_path)
    iterated_set = set()
    test_set = set(test_list)
    val_set = set(val_list)
    for lang1 in new_program_dict.keys():
        for lang2 in new_program_dict.keys():
            if lang2 == lang1:
                continue
            if (lang2, lang1) in iterated_set:
                continue
            iterated_set.add((lang1, lang2))
            file_handler_dic = get_fhs_notok(pair_path, lang1, lang2)
            
            programs1 = new_program_dict[lang1]
            programs2 = new_program_dict[lang2]
            pids1 = pids_dict[lang1]
            pids2 = pids_dict[lang2]
            pids2_set = set(pids2)
            counter_dict = {tag: 0 for tag in tags}
            for pid, program1 in tqdm(zip(pids1, programs1), total=len(pids1)):
                if pid not in pids2_set:
                    continue
                program2 = programs2[pids2.index(pid)]
                if program1 == "" or program2 == "":
                    continue
                tag = 'train'
                if pid in test_set:
                    tag = 'test'
                elif pid in val_set:
                    tag = 'val'
                write_files_notok(tag, program1, program2, pid, file_handler_dic)
                counter_dict[tag] += 1
            for tag in tags:
                for fh in file_handler_dic[tag]:
                    fh.close()
                print(tag, counter_dict[tag])
    return


def get_toked_text(jsons, key='comment', prefix="comment_"):
    failed = []
    for i, js in tqdm(enumerate(jsons)):
        line = js[key]
        text = js[key]
        text_format = text.strip().replace("/*", "")\
                .replace("*/", "").replace("//", "")\
                .replace("\n", " ").replace("'''", "").replace("\"\"\"", "")
        tokens = java_tokenizer(text_format)
        tokens_bpe = bpe_model.apply_bpe(" ".join(tokens))
        js[prefix + 'tokens'] = tokens
        js[prefix + 'bpe'] = tokens_bpe
    return

def get_toked(jsons, tokz, key='snippet'):
    failed = []
    for i, js in enumerate(tqdm(jsons)):
        line = js[key]
        # why this line??
        line = line.replace('\\n', '\n').replace('\\t', '\t')
        try:
            tokens = tokz(line)
            tokens_bpe = bpe_model.apply_bpe(" ".join(tokens))
            js['tokens'] = tokens
            js['bpe'] = tokens_bpe
        except:
            failed.append(js['idx'])
            js['tokens'] = ""
            js['bpe'] = ""
    return failed


def tokenize_data(file_suffix, key, text_keys=[], text_prefixes=[]):
    json_dict = {}
    fail_dict = {}
    for lang, tokz in file_tokenizers.items():
        with open(map_data_path + lang + "-" + file_suffix + ".jsonl") as infile:
            lines = infile.readlines()
            jsons = [json.loads(line.strip()) for line in lines]
            failed_list = get_toked(jsons, tokz, key) #program_formatted program
            for text_key, text_prefix in zip(text_keys, text_prefixes): # comment problem_desc
                get_toked_text(jsons, key=text_key, prefix=text_prefix)
            fail_dict[lang] = failed_list
            json_dict[lang] = jsons

    # Print tokenization failed cased
    for lang in file_tokenizers.keys():
        print(lang, len(fail_dict[lang]))
        print(lang, len(json_dict[lang]))

    # Save tokenized mapping files
    for lang in file_tokenizers.keys():
        jsons = json_dict[lang]
        with jsonlines.open(map_data_path + lang + "-" + file_suffix + "-tok.jsonl", 'w') as outfile:
            outfile.write_all(jsons)
    return


def use_cache(cache_path, file_suffix, key, text_keys=[], text_prefixes=[]):
    json_dict = {}
    fail_dict = {}

    for lang, tokz in file_tokenizers.items():
        cache_dict = {}
        with open(cache_path + lang + "-" + file_suffix + "-tok.jsonl") as infile:
            lines = infile.readlines()
            cached_jsons = [json.loads(line.strip()) for line in lines]
            cache_dict = {js['idx']:js for js in cached_jsons}

        with open(map_data_path + lang + "-" + file_suffix + ".jsonl") as infile:
            lines = infile.readlines()
            jsons = [json.loads(line.strip()) for line in lines]
            in_jsons = []
            out_jsons = []
            for js in jsons:
                if js['idx'] in cache_dict:
                    in_jsons.append(cache_dict[js['idx']])
                else:
                    out_jsons.append(js)
                    
            failed_list = get_toked(out_jsons, tokz, key) #program_formatted program
            for text_key, text_prefix in zip(text_keys, text_prefixes): # comment problem_desc
                get_toked_text(out_jsons, key=text_key, prefix=text_prefix)
            fail_dict[lang] = failed_list
            json_dict[lang] = in_jsons + out_jsons

    # Print tokenization failed cased
    for lang in file_tokenizers.keys():
        print(lang, len(fail_dict[lang]))
        print(lang, len(json_dict[lang]))

    # Save tokenized mapping files
    for lang in file_tokenizers.keys():
        jsons = json_dict[lang]
        with jsonlines.open(map_data_path + lang + "-" + file_suffix + "-tok.jsonl", 'w') as outfile:
            outfile.write_all(jsons)
    return

def get_fhs(pair_path, lang1, lang2, suffix=""):
    path12 = pair_path + lang1 + '-' + lang2 + '/'
    if not os.path.exists(path12):
        os.makedirs(path12)
    
    file_handler_dic = {}
    for tag in tags:
        pth = path12 + tag + '-' + lang1 + '-' + lang2
        fn1 = pth + file_extensions[lang1]
        fn2 = pth + file_extensions[lang2]
        tok1 = pth + '-tok' + file_extensions[lang1]
        tok2 = pth + '-tok' + file_extensions[lang2]
        mp1 = path12 + tag + '-' + lang1 + '-map.jsonl'
        mp2 = path12 + tag + '-' + lang2 + '-map.jsonl'
        outfile1 = open(fn1, 'w')
        outfile2 = open(fn2, 'w')
        outfile_tok1 = open(tok1, 'w')
        outfile_tok2 = open(tok2, 'w')
        outfile11 = open(mp1, 'w')
        outfile22 = open(mp2, 'w')
        file_handler_dic[tag] = [outfile1, outfile2, outfile_tok1, outfile_tok2, outfile11, outfile22]
        if tag != 'train':
            trans_eval1 = pth + '-trans' + file_extensions[lang1]
            trans_eval2 = pth + '-trans' + file_extensions[lang2]
            outfile_trans1 = open(trans_eval1, 'w')
            outfile_trans2 = open(trans_eval2, 'w')
            file_handler_dic[tag] += [outfile_trans1, outfile_trans2]

    return file_handler_dic

def write_files(tag, sn1, sn2, tok1, tok2, mp1, mp2, eval1, eval2, file_handler_dic):
    file_handler_dic[tag][0].write(sn1 + '\n')
    file_handler_dic[tag][1].write(sn2 + '\n')
    file_handler_dic[tag][2].write(tok1 + '\n')
    file_handler_dic[tag][3].write(tok2 + '\n')
    file_handler_dic[tag][4].write(mp1 + '\n')
    file_handler_dic[tag][5].write(mp2 + '\n')
    if len(file_handler_dic[tag]) > 6:
        file_handler_dic[tag][6].write(eval1 + '\n')
        file_handler_dic[tag][7].write(eval2 + '\n')
    return

def get_fhs_notok(pair_path, lang1, lang2):
    path12 = pair_path + lang1 + '-' + lang2 + '/'
    if not os.path.exists(path12):
        os.makedirs(path12)
    file_handler_dic = {}
    for tag in tags:
        pth = path12 + tag + '-' + lang1 + '-' + lang2
        tok1 = pth + '-tok' + file_extensions[lang1]
        tok2 = pth + '-tok' + file_extensions[lang2]
        mp = path12 + tag + '-map.jsonl'
        outfile_tok1 = open(tok1, 'w')
        outfile_tok2 = open(tok2, 'w')
        outfile = open(mp, 'w')
        file_handler_dic[tag] = [outfile_tok1, outfile_tok2, outfile]
    return file_handler_dic
        
def write_files_notok(tag, tok1, tok2, mp, file_handler_dic):
    file_handler_dic[tag][0].write(tok1 + '\n')
    file_handler_dic[tag][1].write(tok2 + '\n')
    file_handler_dic[tag][2].write(mp + '\n')
    return


def get_n_snippets_dict(lang1, lang2, test_set, val_set, n, program_lang_dic, id_lang_dic, recursive=False):
    n_snippets_dict = {tag:[] for tag in tags}
    problem_desc_dic = {}
    for k, cell_set1 in program_lang_dic[lang1].items():
        if k in program_lang_dic[lang2]:
            if k in test_set:
                tag = 'test'
            elif k in val_set:
                tag = 'val'
            else:
                tag = 'train'

            cell_set2 = program_lang_dic[lang2][k]
            cell_list1 = sorted(cell_set1)
            cell_list2 = sorted(cell_set2)

            max_len =  max(cell_list1[-1], cell_list2[-1])
            js1_l = []
            js2_l = []
            for i in range(1, max_len + 1):
                if i not in cell_set1:
                    js1_l.append(None)
                else:
                    js1_l.append(id_lang_dic[lang1][k + '--' + str(i)])
                if i not in cell_set2:
                    js2_l.append(None)
                else:
                    js2_l.append(id_lang_dic[lang2][k + '--' + str(i)])
            nn_list = [n]
            if recursive:
                nn_list = [x for x in range(1, n + 1)]
            if n == -1:
                nn_list = [x for x in range(1, max_len + 1)]
            for nn in nn_list:
                for i in range(0, max_len - nn + 1):
                    idx_l1 = []
                    idx_l2 = []
                    for j in range(0, nn):
                        if js1_l[i + j] == None and js2_l[i + j] == None:
                            continue
                        if nn == 1:
                            if js1_l[i + j] != None and js2_l[i + j] != None:
                                idx = k + '--' + str(i + j + 1)
                                idx_l1.append(idx)
                                idx_l2.append(idx)
                        else:
                            idx = k + '--' + str(i + j + 1)
                            if js1_l[i + j] != None:
                                idx_l1.append(idx)
                            if js2_l[i + j] != None:
                                idx_l2.append(idx)
                    if len(idx_l1) < nn and len(idx_l2) < nn:
                        continue

                    js_l1 = [id_lang_dic[lang1][x] for x in idx_l1]
                    js_l2 = [id_lang_dic[lang2][x] for x in idx_l2]
                    sn1 = " ".join([x['bpe'] for x in js_l1])
                    sn2 = " ".join([x['bpe'] for x in js_l2])
                    if not (sn1.strip() != "" and sn2.strip() != ""):
                        continue
                    tok1 = " ".join([" ".join(x['tokens']) for x in js_l1])
                    tok2 = " ".join([" ".join(x['tokens']) for x in js_l2])
                    if not (tok1.strip() != "" and tok2.strip() != ""):
                        continue
                    id1 = ",".join([x['idx'] for x in js_l1]) 
                    id2 = ",".join([x['idx'] for x in js_l2]) 
                    trans1 = ""
                    trans2 = ""
                    if tag != 'train':
                        pdesc = js_l1[0]['problem_desc']
                        if pdesc in problem_desc_dic:
                            pdtok = problem_desc_dic[pdesc]
                        else:
                            pdtok = java_tokenizer(pdesc)
                            pdtok_bpe = bpe_model.apply_bpe(" ".join(pdtok))
                            problem_desc_dic[pdesc] = pdtok_bpe
                        sid_tok_bpe = pdtok_bpe + ' ' + " ".join(
                            [x['idx'].split('-')[-1] for x in js_l1])
                        trans1 = sid_tok_bpe + " | " + sn1 
                        trans2 = sid_tok_bpe + " | " + sn2
                    n_snippets_dict[tag].append((sn1, sn2, tok1, tok2, id1, id2, trans1, trans2))

    return n_snippets_dict

# generate n-gram snippets
# get_n_snippets(3, test_list, val_list, recursive=True)
def get_n_snippets(n, pair_path, test_list, val_list, program_lang_dic, id_lang_dic, recursive=False):
    pair_path += "pair_data_tok" + '_' + str(n) + '/'
    if recursive:
        pair_path += "pair_data_tok" + '_' + str(n) + 'r/'
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
            file_handler_dic = get_fhs(pair_path, lang1, lang2, n)
            n_snippets_dict = get_n_snippets_dict(lang1, lang2, test_set, val_set, n, program_lang_dic, id_lang_dic, recursive)
            for tag, to_write_list in n_snippets_dict.items():
                for data_to_write in to_write_list:
                    sn1, sn2, tok1, tok2, id1, id2, trans1, trans2 = data_to_write
                    counter_dict[tag] += 1
                    write_files(tag, sn1, sn2,
                                tok1, tok2,
                                id1, id2,
                                trans1, trans2,
                                file_handler_dic)
                for fh in file_handler_dic[tag]:
                    fh.close()
                print(tag, counter_dict[tag])
    return

def read_program_tok_file():
    id_lang_dic = {}
    json_dict = {}
    for lang in file_tokenizers.keys():
        with open(map_data_path + lang + "-program-tok.jsonl") as infile:
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

def read_program_file():
    id_lang_dic = {}
    json_dict = {}
    for lang in file_tokenizers.keys():
        with open(map_data_path + lang + "-program.jsonl") as infile:
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

def get_avg(l):
    return sum(l)/len(l)

def print_stat(label, dic):
    print(label)
    big_l = []
    line = ""
    for lang in langs:
        avg = get_avg(dic[lang])
        line += str(round(avg, 2)) + "\t"
        big_l += dic[lang]
#         print(lang, avg)
    overall_avg = get_avg(big_l)
#     print("Overall", overall_avg)
    line += str(round(overall_avg, 2)) + "\t"
    print(line)
    print()
    return

def print_stat_dict(label, dic):
    print(label)
    line = ""
    sum_v = 0
    for lang in langs:
        sum_v += dic[lang]
        line += str(dic[lang]) + "\t"
    line += str(sum_v) + "\t"
    print(line)
    return

def read_program_tok_file_filtered():
    """
    this is for stats collection. Thus remove the empty lines.
    """
    id_lang_dic = {}
    json_dict = {}
    for lang in file_tokenizers.keys():
        with open(map_data_path + lang + "-program-tok.jsonl") as infile:
            lines = infile.readlines()
            jsons = [json.loads(line.strip()) for line in lines]
            json_dict[lang] = []
            id_dic = {}
            program_dic = {}
            for js in jsons:
                if js['bpe'] == "":
                    continue
                idx_l = js['idx'].split('-')
                pid = idx_l[0]
                id_dic[pid] = js
                json_dict[lang].append(js)
            id_lang_dic[lang] = id_dic
    return json_dict, id_lang_dic


def get_program_new(pair_path, test_list, val_list, program_id_lang_dic):
    pair_path += "pair_data_tok_full/"
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
                    
                    
                    sn1 = js1['bpe']
                    sn2 = js2['bpe']
                    if not (sn1 != "" and sn2 != ""):
                        continue
                    tok1 = " ".join(js1['tokens']) 
                    tok2 = " ".join(js2['tokens'])
                    if not (tok1 != "" and tok2 != ""):
                        continue
                    
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
                        trans1 = sid_tok_bpe + " | " + sn1 
                        trans2 = sid_tok_bpe + " | " + sn2
                    write_files(tag, sn1, sn2,
                                tok1, tok2,
                                k, k,
                                trans1, trans2,
                                file_handler_dic)      
            for tag in tags:
                for fh in file_handler_dic[tag]:
                    fh.close()
                print(tag, counter_dict[tag])
    return


def get_fhs_mono(pair_path, lang1, mono_key, suffix=""):
    path12 = pair_path + lang1 + '-' + mono_key + '/'
    if not os.path.exists(path12):
        os.makedirs(path12)
    
    file_handler_dic = {}
    for tag in tags:
        pth = path12 + tag + '-' + lang1 + '-' + mono_key
        fn1 = pth + file_extensions[lang1]
        fn2 = pth + ".txt"
        tok1 = pth + '-tok' + file_extensions[lang1]
        tok2 = pth + '-tok' + ".txt"
        mp1 = path12 + tag + '-' + lang1 + '-map.jsonl'
        mp2 = path12 + tag + '-' + mono_key + '-map.jsonl'

        outfile1 = open(fn1, 'w')
        outfile2 = open(fn2, 'w')
        outfile_tok1 = open(tok1, 'w')
        outfile_tok2 = open(tok2, 'w')
        outfile11 = open(mp1, 'w')
        outfile22 = open(mp2, 'w')
        file_handler_dic[tag] = [outfile1, outfile2, outfile_tok1, outfile_tok2, outfile11, outfile22]
        if tag != 'train':
            trans_eval1 = pth + '-trans' + file_extensions[lang1]
            trans_eval2 = pth + '-trans' + ".txt"
            outfile_trans1 = open(trans_eval1, 'w')
            outfile_trans2 = open(trans_eval2, 'w')
            file_handler_dic[tag] += [outfile_trans1, outfile_trans2]

    return file_handler_dic

def get_n_snippets_dict_mono(lang1, mono_key, 
                             test_set, val_set, n, program_lang_dic, id_lang_dic, recursive=False):
    n_snippets_dict = {tag:[] for tag in tags}
    problem_desc_dic = {}
    for k, cell_set1 in program_lang_dic[lang1].items():
        if k in test_set:
            tag = 'test'
        elif k in val_set:
            tag = 'val'
        else:
            tag = 'train'

        cell_list1 = sorted(cell_set1)

        max_len =  cell_list1[-1]
        js1_l = []
        for i in range(1, max_len + 1):
            if i not in cell_set1:
                js1_l.append(None)
            else:
                js1_l.append(id_lang_dic[lang1][k + '--' + str(i)])
            
        nn_list = [n]
        if recursive:
            nn_list = [x for x in range(1, n + 1)]
        if n == -1:
            nn_list = [x for x in range(1, max_len + 1)]
        for nn in nn_list:
            for i in range(0, max_len - nn + 1):
                idx_l1 = []
                for j in range(0, nn):
                    if js1_l[i + j] == None:
                        continue
                    if nn == 1:
                        if js1_l[i + j] != None:
                            idx = k + '--' + str(i + j + 1)
                            idx_l1.append(idx)
                    else:
                        idx = k + '--' + str(i + j + 1)
                        if js1_l[i + j] != None:
                            idx_l1.append(idx)
                if len(idx_l1) < nn:
                    continue

                js_l1 = [id_lang_dic[lang1][x] for x in idx_l1]
                sn1 = " ".join([x['bpe'] for x in js_l1])
                sn2 = " ".join([x[mono_key + '_bpe'] for x in js_l1])
                if mono_key == 'desc':
                    sn2 = x[mono_key + '_bpe']
                if not (sn1 != "" and sn2 != ""):
                    continue
                tok1 = " ".join([" ".join(x['tokens']) for x in js_l1])
                tok2 = " ".join([" ".join(x[mono_key + '_tokens']) for x in js_l1])
                if mono_key == 'desc':
                    tok2 = " ".join(x[mono_key + '_tokens'])
                id1 = ",".join([x['idx'] for x in js_l1]) 
                id2 = id1
                trans1 = ""
                trans2 = ""
                if tag != 'train':
                    pdesc = js_l1[0]['problem_desc']
                    if pdesc in problem_desc_dic:
                        pdtok = problem_desc_dic[pdesc]
                    else:
                        pdtok = java_tokenizer(pdesc)
                        pdtok_bpe = bpe_model.apply_bpe(" ".join(pdtok))
                        problem_desc_dic[pdesc] = pdtok_bpe
                    sid_tok_bpe = pdtok_bpe + ' ' + " ".join(
                        [x['idx'].split('-')[-1] for x in js_l1])
                    trans1 = sid_tok_bpe + " | " + sn1 
                    trans2 = sid_tok_bpe + " | " + sn2
                n_snippets_dict[tag].append((sn1, sn2, tok1, tok2, id1, id2, trans1, trans2))

    return n_snippets_dict

# generate n-gram snippets
# get_n_snippets(3, test_list, val_list, recursive=True)
def get_n_snippets_mono(n, pair_path, mono_key, test_list, val_list, program_lang_dic, id_lang_dic, recursive=False):
    pair_path += "pair_data_tok_" + str(n) + "_" + mono_key + '/'
    if recursive:
        pair_path += "pair_data_tok_" + str(n)  + "_" + mono_key + '_r/'
    if not os.path.exists(pair_path):
        os.makedirs(pair_path)

    iterated_set = set()
    test_set = set(test_list)
    val_set = set(val_list)
    all_set = test_set | val_set
    problem_desc_dic = {}
    for lang1 in file_tokenizers.keys():
        print(lang1)
        counter_dict = {tag: 0 for tag in tags}
        file_handler_dic = get_fhs_mono(pair_path, lang1, mono_key, n)
        n_snippets_dict = get_n_snippets_dict_mono(lang1, mono_key, 
                                              test_set, val_set, n, 
                                              program_lang_dic, id_lang_dic, recursive)
        for tag, to_write_list in n_snippets_dict.items():
            for data_to_write in to_write_list:
                sn1, sn2, tok1, tok2, id1, id2, trans1, trans2 = data_to_write
                counter_dict[tag] += 1
                write_files(tag, sn1, sn2,
                            tok1, tok2,
                            id1, id2,
                            trans1, trans2,
                            file_handler_dic)
            for fh in file_handler_dic[tag]:
                fh.close()
            print(tag, counter_dict[tag])
    return

def get_program_new_mono(pair_path, mono_key, test_list, val_list, program_id_lang_dic):
    pair_path += "pair_data_tok_full" + "_" + mono_key + '/'
    if not os.path.exists(pair_path):
        os.makedirs(pair_path)

    iterated_set = set()
    test_set = set(test_list)
    val_set = set(val_list)
    all_set = test_set | val_set
    problem_desc_dic = {}
    for lang1 in file_tokenizers.keys():
        print(lang1)
        counter_dict = {tag: 0 for tag in tags}
        file_handler_dic = get_fhs_mono(pair_path, lang1, mono_key, 'full')

        for k, js1 in program_id_lang_dic[lang1].items():
            if k not in all_set:
                tag = 'train'
            elif k in test_set:
                tag = 'test'
            elif k in val_set:
                tag = 'val'
            
            sn1 = js1['bpe']
            sn2 = js1[mono_key + '_bpe']
            if not (sn1 != "" and sn2 != ""):
                continue
            tok1 = " ".join(js1['tokens'])
            tok2 = " ".join(js1[mono_key + '_tokens'])
            if not (tok1 != "" and tok2 != ""):
                continue
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
                trans1 = sid_tok_bpe + " | " + sn1
                trans2 = sid_tok_bpe + " | " + sn2
            write_files(tag, sn1, sn2,
                        tok1, tok2,
                        k, k,
                        trans1, trans2,
                        file_handler_dic)      
        for tag in tags:
            for fh in file_handler_dic[tag]:
                fh.close()
            print(tag, counter_dict[tag])
    return

def get_program_new_mono_comment(pair_path, mono_key, test_list, val_list, id_lang_dic, program_id_lang_dic):
    pair_path += "pair_data_tok_full_desc_comment/"
    if not os.path.exists(pair_path):
        os.makedirs(pair_path)

    iterated_set = set()
    test_set = set(test_list)
    val_set = set(val_list)
    all_set = test_set | val_set
    problem_desc_dic = {}
    for lang1 in file_tokenizers.keys():
        print(lang1)
        counter_dict = {tag: 0 for tag in tags}
        file_handler_dic = get_fhs_mono(pair_path, lang1, mono_key, 'full')

        for k, js1 in program_id_lang_dic[lang1].items():
            if k not in all_set:
                tag = 'train'
            elif k in test_set:
                tag = 'test'
            elif k in val_set:
                tag = 'val'
            
            c_bpe_list = []
            c_tokens_list = []
            sn_ids = js1['snippet_ids']
            for sid in sn_ids:
                sn_id = k + '--' + str(sid)
                sn = id_lang_dic[lang1][sn_id]
                c_bpe = sn['comment_bpe']
                c_bpe_list.append(c_bpe)
                c_tokens = sn['comment_tokens']
                c_tokens_list.append(" ".join(c_tokens))
            
            comment_bpe = ' ; '.join(c_bpe_list)
            comment_tokens = ' ; '.join(c_tokens_list)
            sn1 = js1['bpe']
            sn2 = js1['desc_bpe'] + " | " + comment_bpe
            if not (sn1 != "" and sn2 != ""):
                continue
            tok1 = " ".join(js1['tokens'])
            tok2 = " ".join(js1['desc_tokens']) + " | " + comment_tokens
            if not (tok1 != "" and tok2 != ""):
                continue
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
                trans1 = sid_tok_bpe + " | " + sn1
                trans2 = sid_tok_bpe + " | " + sn2
            write_files(tag, sn1, sn2,
                        tok1, tok2,
                        k, k,
                        trans1, trans2,
                        file_handler_dic)      
        for tag in tags:
            for fh in file_handler_dic[tag]:
                fh.close()
            print(tag, counter_dict[tag])
    return

# bpe_model, dico = get_bpe()
