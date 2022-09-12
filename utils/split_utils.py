from common_utils import *

def get_eval_list(split_dict):
    test_keys = set()
    valid_keys = set()
    for lang in langs:
        test_keys = test_keys | set(split_dict[lang]['test'])
        valid_keys = valid_keys | set(split_dict[lang]['valid'])
    test_list_num = sorted([int(x) for x in test_keys])
    test_list = [str(x) for x in test_list_num]
    valid_list_num = sorted([int(x) for x in valid_keys])
    valid_list = [str(x) for x in valid_list_num]
    return test_list, valid_list


def save_split_dict(split_dict):
    with open(split_dict_path, 'w') as outfile:
        json.dump(split_dict, outfile)
    return
    
def load_split_dict():
    with open(split_dict_path) as infile:
        split_dict = json.load(infile) 
    return split_dict

def get_all_desc_dict(program_id_lang_dic):
    desc_dict = {}
    for lang in langs:
        for k, v in program_id_lang_dic[lang].items():
            desc = v['problem_desc']
            if desc in desc_dict:
                desc_dict[desc].add(k)
            else:
                desc_dict[desc] = set([k])
    for k, v in desc_dict.items():
        desc_dict[k] = list(v)
    return desc_dict

def get_desc_dict(program_id_lang_dic, lang):
    desc_dict = {}
    for k, v in program_id_lang_dic[lang].items():
        desc = v['problem_desc']
        if desc in desc_dict:
            desc_dict[desc].append(k)
        else:
            desc_dict[desc] = [k]
    return desc_dict

def get_sim_dict(sim_dict_path, desc_list):
    if os.path.exists(sim_dict_path):
        with open(sim_dict_path, 'rb') as infile:
            sim_dict = pickle.load(infile)
        return sim_dict
    sim_dict = {}
    for i, desc1 in enumerate(tqdm(desc_list)):
        desc1_list = desc1.split()
        len1 = len(desc1_list)
        for j, desc2 in enumerate(desc_list):
            if j > i:
                desc2_list = desc2.split()
                len2 = len(desc2_list)
                if abs(len1-len2) > 2:
                    continue
                sm = difflib.SequenceMatcher(None,desc1_list,desc2_list)
                sim_dict[(i, j)] = sm.ratio()
    with open(sim_dict_path, 'wb') as outfile:
        pickle.dump(sim_dict, outfile)
    return sim_dict

def get_desc_similarity(sim_dict, desc_list):
    sim_desc_dict = {}
    for k, v in tqdm(sim_dict.items()):
        thres = 0.7
        desc1 = desc_list[k[0]]
        desc2 = desc_list[k[1]]
        len1 = len(desc1.split())
        len2 = len(desc2.split())
        if min(len1, len2) < 5:
            thres = 0.6
        elif max(len1, len2) > 10:
            thres = 0.8
        else:
            thres = 0.7
        if v > thres:
            if desc1 in sim_desc_dict:
                sim_desc_dict[desc1].add(desc2)
            else:
                sim_desc_dict[desc1] = set([desc2])
    return sim_desc_dict

def group_similar_desc(sim_desc_dict):
    new_sim_desc_dict = {}
    skip_set = set()
    for k, v in sim_desc_dict.items():
        if k in skip_set:
            continue
        new_sim_desc_dict[k] = sim_desc_dict[k]
        set_v = set(v)
        for desc in set_v:
            skip_set.add(desc)
            if desc in sim_desc_dict:
                new_sim_desc_dict[k] =  new_sim_desc_dict[k] | sim_desc_dict[desc]
    return new_sim_desc_dict

def group_similar_program_id(desc_dict, new_sim_desc_dict):
    id_group_list = []
    skip_set = set()
    for k, v in desc_dict.items():
        if k in skip_set:
            continue
        map_list = desc_dict[k]
        if k in new_sim_desc_dict:
            for desc in new_sim_desc_dict[k]:
                map_list += desc_dict[desc]
                skip_set.add(desc)
        id_group_list.append(map_list)  
    return id_group_list

def get_union_all_key_list(program_id_lang_dic, id_group_list):
    common_keys_7 = get_common_keys(langs, program_id_lang_dic)
    common_keys_6 = get_common_keys(langs[:-1], program_id_lang_dic)
    common_keys_5 = get_common_keys(langs[:-2], program_id_lang_dic)

    key_mapping_dict = {}
    new_keys_list = []
    for l in id_group_list:
        k5, k6, k7 = -1,-1,-1
        for key in l:
            if key in set(common_keys_7):
                k7 = key
            elif key in set(common_keys_6):
                k6 = key
            elif key in set(common_keys_5):
                k5 = key
        if k7 != -1:
            new_keys_list.append(k7)
        elif k6 != -1:
            new_keys_list.append(k6)
        elif k5 != -1:
            new_keys_list.append(k5)
        else:
            new_keys_list.append(l[0])
        key_mapping_dict[new_keys_list[-1]] = l

    return new_keys_list, key_mapping_dict

def get_union_program_lang_dic(program_id_lang_dic, new_keys_list):
    union_program_lang_dic = {}
    for lang in langs:
        keys = program_id_lang_dic[lang].keys()
        a = set(keys) & set(new_keys_list)
        keys_list = list(a)
        union_program_lang_dic[lang] = {x:[] for x in keys_list}
    return union_program_lang_dic

def expand_union_key_lists(test_list, key_mapping_dict):
    new_test_list = []
    for idx in test_list:
        if idx in key_mapping_dict:
            new_test_list += key_mapping_dict[idx]
        else:
            new_test_list.append(idx)
    return new_test_list

def expand_union_split_dict(split_dict_union, key_mapping_dict):
    split_dict = {}
    for lang in langs:
        split_dict[lang] = {}
        for tag in split_dict_union[lang].keys():
            key_list = split_dict_union[lang][tag]
            new_ket_list = expand_union_key_lists(key_list, key_mapping_dict)
            split_dict[lang][tag] = new_ket_list
    return split_dict

def show_split_info(split_dict):
    for lang in langs:
        len_val = len(split_dict[lang]['valid'])
        len_test = len(split_dict[lang]['test'])
        len_train = len(split_dict[lang]['train'])
        sums = len_val + len_test + len_train
        print(lang, len_val, len_test, len_train)
        print(lang, len_val/sums, len_test/sums, len_train/sums)
    return

def get_split(split_param_test, split_param_valid):
    '''优先使用common keys作为split
        Constraints:
        New test should include previous test, and exclude previous train
        New train should include previous train
        Don't include keys from previous set, when it's not in the current set'''
    split_dict = {lang:{} for lang in langs}
    test_keys = set()
    valid_keys = set()
    train_keys = set()
    for lang in langs[::-1]:
        k, v = lang, program_lang_dic[lang]
        lang_all_keys = set(list(v.keys()))
        # remove previous test key
        lang_test_keys = lang_all_keys & test_keys
        lang_res_test_keys = lang_all_keys - lang_test_keys
        # remove previous train key
        lang_train_keys = lang_res_test_keys & train_keys
        lang_res_train_keys = lang_res_test_keys - lang_train_keys
        # remove previous valid key
        lang_valid_keys = lang_res_train_keys & valid_keys
        lang_new_keys = lang_res_train_keys - lang_valid_keys

        lang_new_keys_common_6 = lang_new_keys & set(common_keys_6)
        lang_res_keys_common_6 = lang_new_keys - lang_new_keys_common_6

        n = int(split_param_test * len(v))
        m = int(split_param_valid * len(v))
        if len(test_keys) == 0:
            random.shuffle(common_keys)
            X_test = common_keys[:n]
            X_valid = common_keys[n:n+m]
        else:
            if len(lang_new_keys) <= n+m-len(lang_test_keys) - len(lang_valid_keys):
                X_test = list(lang_test_keys)
                X_valid = list(lang_valid_keys)
            else:
    #             优先选择common keys
                lang_new_keys_common_6_list = list(lang_new_keys_common_6)
                lang_res_keys_common_6_list = list(lang_res_keys_common_6)
                random.shuffle(lang_new_keys_common_6_list)
                random.shuffle(lang_res_keys_common_6_list)
                lang_new_keys_list = lang_new_keys_common_6_list + lang_res_keys_common_6_list
    #             lang_new_keys_list = list(lang_new_keys)
    #             random.shuffle(lang_new_keys_list)
                X_test = lang_new_keys_list[:n-len(lang_test_keys)] + list(lang_test_keys)
                X_valid = lang_new_keys_list[n-len(lang_test_keys):
                                          n+m-len(lang_test_keys) - len(lang_valid_keys)] + list(lang_valid_keys)
        X_train = list(lang_all_keys - set(X_test) - set(X_valid))
        split_dict[lang]['test'] = X_test
        split_dict[lang]['valid'] = X_valid
        split_dict[lang]['train'] = X_train
        assert(len(set(X_test) & valid_keys) == 0 and len(set(X_test) & train_keys) == 0)
        assert(len(set(X_valid) & test_keys) == 0 and len(set(X_valid) & train_keys) == 0)
        assert(len(set(X_train) & test_keys) == 0 and len(set(X_train) & valid_keys) == 0)
        test_keys = test_keys | set(X_test)
        valid_keys = valid_keys | set(X_valid)
        train_keys = train_keys | set(X_train)
        assert(len(test_keys & valid_keys) == 0)
        assert(len(test_keys & train_keys) == 0)
        assert(len(valid_keys & train_keys) == 0)
        a = len(split_dict[lang]['test'])+ len(split_dict[lang]['valid'])+ len(split_dict[lang]['train'])
        assert(a == len(v))
        print(len(split_dict[lang]['test']), len(split_dict[lang]['valid']), len(split_dict[lang]['train']), a)
        print(len(split_dict[lang]['test'])/a, len(split_dict[lang]['valid'])/a, len(split_dict[lang]['train'])/a)
    return split_dict

def get_all_keys(langs, program_lang_dic):
    all_keys = set()
    for lang in langs:
        k, v = lang, program_lang_dic[lang]
        keys = list(v.keys())
        all_keys = all_keys | set(keys)
    return all_keys

def get_common_keys(langs, program_lang_dic):
    common_keys = []
    print(langs)
    for lang in langs:
        k, v = lang, program_lang_dic[lang]
        if len(common_keys) == 0:
            common_keys = v.keys()
        common_keys = list(set(v.keys()) & set(common_keys))
    return common_keys

def get_split_random(split):
    '''Split randomly without taking common_keys into consideration'''
    split_dict = {lang:{} for lang in langs}
    test_keys = set()
    valid_keys = set()
    train_keys = set()
    for lang in langs[::-1]:
        k, v = lang, program_lang_dic[lang]
        lang_all_keys = set(list(v.keys()))
        print(len(v))
        # remove previous test key
        lang_test_keys = lang_all_keys & test_keys
        lang_res_test_keys = lang_all_keys - lang_test_keys
        # remove previous train key
        lang_train_keys = lang_res_test_keys & train_keys
        lang_res_train_keys = lang_res_test_keys - lang_train_keys
        # remove previous valid key
        lang_valid_keys = lang_res_train_keys & valid_keys
        lang_new_keys = lang_res_train_keys - lang_valid_keys

        if len(lang_new_keys) < 10:
            split_dict[lang]['test'] = list(lang_test_keys)
            split_dict[lang]['valid'] = list(lang_valid_keys)
            split_dict[lang]['train'] = list(lang_res_test_keys - lang_valid_keys)
        else:
            # num of test = 0.1 * n
            # new_n = n - len(lang_train_keys)
            # x * new_n + len(lang_test_keys) = 0.1 * n
            # x = (0.1 * len(v) - len(lang_test_keys))/ (len(v) - len(lang_train_keys))
            split_param_test =  (split * len(v) - len(lang_test_keys))/ (len(v) - len(lang_train_keys))
            if split_param_test < 0:
                print("test", split_param)
            # num of test = 0.1 * n
            # new_n = n - len(lang_train_keys) - len(X_test)
            # x * new_n + len(lang_valid_keys) = 0.1 * n
            # x = (0.1 * len(v) - len(lang_valid_keys))/ (len(v) - len(lang_train_keys)- len(X_test))
            split_param_valid =  (split * len(v) - len(lang_valid_keys))/ (len(v) - 
                                                                           len(lang_train_keys) - len(X_test))
            if split_param_valid < 0:
                print("valid", split_param)
            X_train, X_test= train_test_split(list(lang_new_keys), test_size=split_param_test, random_state=42)
            X_train, X_valid= train_test_split(list(X_train), test_size=split_param_valid, random_state=42)

            split_dict[lang]['test'] = X_test + list(lang_test_keys)
            split_dict[lang]['valid'] = X_valid + list(lang_valid_keys)
            split_dict[lang]['train'] = X_train + list(lang_train_keys)
            test_keys = test_keys | set(X_test)
            valid_keys = valid_keys | set(X_valid)
            train_keys = train_keys | set(X_train)
    #     print(len(test_keys & valid_keys))
    #     print(len(test_keys & train_keys))
    #     print(len(valid_keys & train_keys))
        a = len(split_dict[lang]['test'])+ len(split_dict[lang]['valid'])+ len(split_dict[lang]['train'])
        print(len(split_dict[lang]['test']), len(split_dict[lang]['valid']), len(split_dict[lang]['train']), a)
    return split_dict

def get_split_common_keys(split_param_test, split_param_valid, program_lang_dic):
    '''使用common keys作为split
    Strategy: 
    C的test/valid从7种语言的common keys中选
    PHP的test/valid先算上C的test/valid，然后从6种语言的common keys（去掉C的所有keys）中选
    其它语言的test/valid先算上PHP和C的test/valid，然后从5种语言的common keys（去掉C和PHP的所有keys）中选
    前五种语言的eval set包含PHP和C的，同时不与它们的train重叠
    相当于从总共的7928个file中，选出800个做test，800个做valid，这1600个eval files全部来自common_5
    这1600个eval files包含460个common_6和全部的common_7
    common_5: ++++++++++++++++|*******|&&
    eval_files: ++++++|***|&&
    使用common keys可以用最少的file来保证每种语言的eval

    '''
    if os.path.exists(split_dict_path):
        print("split_dict already exists! Load from cache.")
        return load_split_dict()
    split_dict = {lang:{} for lang in langs}
    test_keys = set()
    valid_keys = set()
    train_keys = set()
    all_keys = set()
    all_keys = get_all_keys(langs, program_lang_dic)
    test_n_all = int(split_param_test * len(all_keys))
    valid_n_all = int(split_param_valid * len(all_keys))
    print(test_n_all, valid_n_all)
#     test_n = 800
#     valid_n = 800
    common_keys_7 = get_common_keys(langs, program_lang_dic)
    common_keys_6 = get_common_keys(langs[:-1], program_lang_dic)
    common_keys_5 = get_common_keys(langs[:-2], program_lang_dic)
    print("5", len(common_keys_5), '6', len(common_keys_6))
    
    random.shuffle(common_keys_5)
    random.shuffle(common_keys_6)
    random.shuffle(common_keys_7)
    n_c_test = len(common_keys_7)//2

    c_keys = set(list(program_lang_dic["C"].keys()))
    php_keys = set(list(program_lang_dic["PHP"].keys()))
    common_test_keys = set()
    common_valid_keys = set()

    for lang in langs[::-1]:
        k, v = lang, program_lang_dic[lang]
        v_keys_list = list(v.keys())
        random.shuffle(v_keys_list)
        lang_all_keys = set(v_keys_list)
        all_keys = all_keys | lang_all_keys
        test_n = test_n_all
        valid_n = valid_n_all
        if lang == "C":
            # C的test和valid只能从common_key_7中取
            test_n = int(len(v) * split_param_test)
            valid_n = int(len(v) * split_param_valid)
            if valid_n < 40:
                valid_n = 40
#             if test_n < valid_n * 2:
#                 test_n = valid_n * 2
            X_test = common_keys_7[:test_n]
            X_valid = common_keys_7[test_n:test_n+valid_n]
            X_train = list(c_keys - set(X_test) - set(X_valid))
        else:
            
            lang_res_keys = set()
            X_test = list(test_keys)
            X_valid = list(valid_keys)
            if lang == "PHP":
                test_n = int(len(v) * split_param_test)
                valid_n = int(len(v) * split_param_valid)
                if (len(X_test) < test_n) | (len(X_valid) < valid_n):
                    # C的test和valid只能从common_key_6中取
                    lang_res_keys = (lang_all_keys - c_keys
                                     - set(X_test) - set(X_valid)) & set(common_keys_6)
            else:
                if (len(X_test) < test_n) | (len(X_valid) < valid_n):
                    # 其它语言的test和valid只能从common_key_5中取
                    lang_res_keys = (lang_all_keys - c_keys - php_keys
                                     - set(X_test) - set(X_valid)) & set(common_keys_5)
            if len(lang_res_keys) > 0:
                X_test = list(lang_res_keys)[:test_n-len(X_test)] + X_test
                X_valid = list(lang_res_keys - set(X_test))[:valid_n-len(X_valid)] + X_valid
            X_train = list(lang_all_keys - set(X_test) - set(X_valid))
        split_dict[lang]['test'] = X_test
        split_dict[lang]['valid'] = X_valid
        split_dict[lang]['train'] = X_train

        assert(len(set(X_test) & set(X_valid)) == 0 and len(set(X_test) & set(X_train)) == 0)
        assert(len(set(X_train) & set(X_valid)) == 0)
        
        assert(len(set(X_test) & valid_keys) == 0 and len(set(X_test) & train_keys) == 0)
        assert(len(set(X_valid) & test_keys) == 0 and len(set(X_valid) & train_keys) == 0)
        assert(len(set(X_train) & test_keys) == 0 and len(set(X_train) & valid_keys) == 0)
        test_keys = test_keys | set(X_test)
        valid_keys = valid_keys | set(X_valid)
        train_keys = train_keys | set(X_train)
        assert(len(test_keys & valid_keys) == 0)
        assert(len(test_keys & train_keys) == 0)
        assert(len(valid_keys & train_keys) == 0)
        assert(len(test_keys & valid_keys) == 0)
        a = len(split_dict[lang]['test'])+ len(split_dict[lang]['valid'])+ len(split_dict[lang]['train'])
        assert(a == len(v))
        print(lang, len(split_dict[lang]['test']), 
              len(split_dict[lang]['valid']), len(split_dict[lang]['train']))
        print(lang, len(split_dict[lang]['test'])/a, 
              len(split_dict[lang]['valid'])/a, len(split_dict[lang]['train'])/a)
    b = len(test_keys)+ len(valid_keys)+ len(train_keys)
    assert(b == len(all_keys))
    print(len(test_keys), len(valid_keys), len(train_keys))
    print(len(test_keys)/b, len(valid_keys)/b, len(train_keys)/b)
    return split_dict
                       
def test_get_split_common_keys(split_param_test, split_param_valid, program_lang_dic):
    split_dict = get_split_common_keys(split_param_test, split_param_valid, program_lang_dic)
    c_eval = set(split_dict['C']['test']) | set(split_dict['C']['valid'])
    php_eval = set(split_dict['PHP']['test']) | set(split_dict['PHP']['valid'])
    js_eval = set(split_dict['Javascript']['test']) | set(split_dict['Javascript']['valid'])
    cs_eval = set(split_dict['C#']['test']) | set(split_dict['C#']['valid'])
    test_keys = set()
    valid_keys = set()
    for lang in langs:
        test_keys = test_keys | set(split_dict[lang]['test'])
        valid_keys = valid_keys | set(split_dict[lang]['valid'])
    all_eval = test_keys|valid_keys

    print(len(c_eval), len(php_eval), len(c_eval & php_eval))
    print(len(php_eval), len(js_eval), len(js_eval & php_eval))
    print(len(js_eval), len(cs_eval), len(js_eval & cs_eval))
    print(len(all_eval), len(js_eval & all_eval))
    return 
