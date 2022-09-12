from tokenization_utils import *
# Tokenize codeXglue
def get_toked_lines(lines, tokz, line_num=False):
    failed = []
    tokens_list = []
    tokens_bpe_list = []
    for i, line in tqdm(enumerate(lines)):
        line = line.strip().replace('\\n', '\n').replace('\\t', '\t')
        if line_num:
            line = str(i) + " | " + line
        tokens = ""
        tokens_bpe = ""
        try:
            tokens = tokz(line)
            tokens_bpe = bpe_model.apply_bpe(" ".join(tokens))
        except:
            failed.append(i)
        tokens_list.append(" ".join(tokens))
        tokens_bpe_list.append(tokens_bpe)
    return tokens_list, tokens_bpe_list, failed

cxg_data_path = "/home/mingzhu/CodeModel/g4g/codeXglue_data_new/"
cxg_tags = ['train', 'test', 'valid']
langs = ["Java", "C#"]

fns = os.listdir(cxg_data_path)
for fn in fns:
    lang = "Java"
    if fn.endswith('txt.cs'):
        lang = "C#"
        tokenizer = file_tokenizers['C#']
    elif fn.endswith('txt.java'):
        tokenizer = file_tokenizers['Java']
    else:
        continue
    with open(cxg_data_path + fn) as infile:
        lines = infile.readlines()
    line_num = True
    tag = fn.split(".")[0]
    if tag == "train":
        line_num = False
    
    tokens_list, tokens_bpe_list, failed = get_toked_lines(lines, tokenizer, line_num)
    tok_fn = tag + "-" + "Java-C#-tok" + file_extensions[lang]
    bpe_fn = tag + "-" + "Java-C#" + file_extensions[lang]
    if tag != "train":
        bpe_fn = tag + "-" + "Java-C#-trans" + file_extensions[lang]
    with open(cxg_data_path + tok_fn, 'w') as outfile1, \
        open(cxg_data_path + bpe_fn, 'w') as outfile2:
        for tok, bpe in zip(tokens_list, tokens_bpe_list):
            outfile1.write(tok + "\n")
            outfile2.write(bpe + "\n")
        
path = cxg_data_path
fns = os.listdir(cxg_data_path)
for fn in fns:
    if fn.endswith(".pth"):
        os.remove(os.path.join(path, fn))
lang1 = "Java"
lang2 = "C#"
# train.java-cs.txt.bpe.cs 
# test.java-cs.txt.tok.java
for tag in tags:
    if tag == 'val':
        tag = 'valid'
    lang1_lower = lang_lower[lang1]
    lang2_lower = lang_lower[lang2]
    fn_prefix = path + tag + "-" + lang1 + "-" + lang2
    fn1 = fn_prefix + file_extensions[lang1]
    fn2 = fn_prefix + file_extensions[lang2]
    print(fn1)
    print(fn2)
    if tag != 'train':
        fn1 = fn_prefix + "-trans" + file_extensions[lang1]
        fn2 = fn_prefix + "-trans" + file_extensions[lang2]
    tag_pth = tag
    if tag == 'val':
        tag_pth = 'valid'
    fn_pth_prefix = path + tag_pth + "." + lang1_lower + "_sa-" + lang2_lower + "_sa."
    fn1_pth = fn_pth_prefix + lang1_lower + "_sa.pth"
    fn2_pth = fn_pth_prefix + lang2_lower + "_sa.pth"
    fn_pth_prefix_alt = path + tag_pth + "." + lang2_lower + "_sa-" + lang1_lower + "_sa."
    fn1_pth_alt = fn_pth_prefix_alt + lang1_lower + "_sa.pth"
    fn2_pth_alt = fn_pth_prefix_alt + lang2_lower + "_sa.pth"

    XLM_preprocess(voc_path, fn1, fn1_pth)
    XLM_preprocess(voc_path, fn2, fn2_pth)
    copyfile(fn1_pth, fn1_pth_alt)
    copyfile(fn2_pth, fn2_pth_alt)