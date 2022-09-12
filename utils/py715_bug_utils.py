from common_utils import *

def fix_mapping_py715(file_suffix, faulty_key, key):
    lang = 'Python'
    with open(map_data_path + lang + "-" + file_suffix + "-tok.jsonl") as infile:
        lines = infile.readlines()
        jsons = [json.loads(line.strip()) for line in lines]

    for i, dic in enumerate(jsons):
        if dic['idx'] == faulty_key:
            tokens = file_tokenizers[lang](dic[key])
            tokens_bpe = bpe_model.apply_bpe(" ".join(tokens))
            jsons[i]['tokens'] = tokens
            jsons[i]['bpe'] = tokens_bpe
            pass

    with jsonlines.open(map_data_path + lang + "-" + file_suffix + "-tok.jsonl", 'w') as outfile:
        outfile.write_all(jsons)
    return

def get_fixed_py715(file_suffix, faulty_key):
    lang = 'Python'
    with open(map_data_path + lang + "-" + file_suffix + "-tok.jsonl") as infile:
        lines = infile.readlines()
        jsons = [json.loads(line.strip()) for line in lines]

    for i, dic in enumerate(jsons):
        if dic['idx'] == faulty_key:
            return " ".join(dic['tokens'])
    return ""

def check_pairdata_misalignment():
    datasets = os.listdir(data_path)
    for dataset in datasets:
        if not dataset.startswith('pair'):
            continue
        dataset_path = data_path + "/" + dataset
        lang_folders = os.listdir(dataset_path)
        for lang_folder in lang_folders:
            lang_path = dataset_path + "/" + lang_folder + '/'
            lang1, lang2 = lang_folder.split('-')
            lang_pair = lang_folder
            print(dataset, lang_pair)
            for tag in tags:
                fn1 = tag + '-' + lang_pair + '-tok' + file_extensions[lang1]
                fn2 = tag + '-' + lang_pair + '-tok' + file_extensions[lang2]
                line_count1 = 0
                line_count2 = 0
                with open(lang_path + fn1) as infile:
                    line_count1 = len(infile.readlines())
                with open(lang_path + fn2) as infile:
                    line_count2 = len(infile.readlines())
                if line_count1 != line_count2:
                    print(dataset, lang_pair, tag, line_count1, line_count2)
    return
            
# TODO: 区分snippet和program
def fix_pair_data_py715(correct_code, check_program=True):
    lang_py = "Python"
    datasets = os.listdir(data_path)
    for dataset in datasets:
        if not dataset.startswith('pair'):
            continue
        is_program = False
        faulty_id = "715-Python-12"
        if "full" in dataset:
            is_program = True
            faulty_id = "715"
        if is_program != check_program:
            continue
        dataset_path = data_path + "/" + dataset
        lang_folders = os.listdir(dataset_path)
        for lang_folder in lang_folders:
            if lang_py in lang_folder:
                
                lang_path = dataset_path + "/" + lang_folder + '/'
                lang1, lang2 = lang_folder.split('-')
                lang_pair = lang_folder
                for tag in tags:
                    py_fn = tag + '-' + lang_pair + '-tok' + file_extensions[lang_py]
                    py_map = tag + '-' + lang_py + '-map.jsonl'
                    faulty_line_num = 0
                    with open(lang_path + py_map) as infile:
                        map_lines = infile.readlines()
                        for i, line in enumerate(map_lines):
                            num = line.strip()
                            if num == faulty_id:
                                faulty_line_num = i
                                break
                    print(dataset, lang_pair, tag)
                    if faulty_line_num != 0:
                        with open(lang_path + py_fn) as infile:
                            lines = infile.readlines()
                        faulty_lines = lines[faulty_line_num:faulty_line_num+3]
                        lines[faulty_line_num] = correct_code + '\n'
                        lines.pop(faulty_line_num+1)
                        # after one pop, the index for the next pop has changed. Thus we use the same index
                        lines.pop(faulty_line_num+1)
                        with open(lang_path + py_fn, 'w') as outfile:
                            for line in lines:
                                outfile.write(line)
    return