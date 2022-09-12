from common_utils import *
from code_prepro.data.dictionary import Dictionary


# from CodeGen/codegen_sources/model/preprocess.py
def XLM_preprocess(voc_path, txt_path, bin_path):
    assert os.path.isfile(voc_path)
    assert os.path.isfile(txt_path)
    dico = Dictionary.read_vocab(voc_path)
    data = Dictionary.index_data(txt_path, bin_path, dico)
    print(
        "%i words (%i unique) in %i sentences."
        % (
            len(data["sentences"]) - len(data["positions"]),
            len(data["dico"]),
            len(data["positions"]),
        )
    )
    if len(data["unk_words"]) > 0:
        print(
            "%i unknown words (%i unique), covering %.2f%% of the data."
            % (
                sum(data["unk_words"].values()),
                len(data["unk_words"]),
                sum(data["unk_words"].values())
                * 100.0
                / (len(data["sentences"]) - len(data["positions"])),
            )
        )
        if len(data["unk_words"]) < 30000:
            for w, c in sorted(data["unk_words"].items(), key=lambda x: x[1])[::-1][
                :30
            ]:
                print("%s: %i" % (w, c))
    return
# Binarize the tokenized files
# binarize("/home/mingzhu/CodeModel/g4g/pair_data_tok_3r/", file_extensions.keys(), "data/bpe/cpp-java-python/vocab")
def binarize(root, langs, voc_path):
    iterated_set = set()
    for lang1 in langs:
        for lang2 in langs:
            if lang2 == lang1:
                continue
            if (lang2, lang1) in iterated_set:
                continue
            iterated_set.add((lang1, lang2))
            print(lang1, lang2)
            lang1_lower = lang_lower[lang1]
            lang2_lower = lang_lower[lang2]

            path = root + lang1 + '-' + lang2 + '/'
            fns = os.listdir(path)
            for fn in fns:
                if fn.endswith(".pth"):
                    os.remove(os.path.join(path, fn))
            for tag in tags:
                fn_prefix = path + tag + "-" + lang1 + "-" + lang2
                fn1 = fn_prefix + file_extensions[lang1]
                fn2 = fn_prefix + file_extensions[lang2]
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



def read_cont_results(exp_prefix, dump_path, 
                 print_result=True, print_em=False, print_copy=False, 
                 hypo_collection_path=None):
    beam_size = 5
    fns = os.listdir(dump_path)
    for fn in fns:
        if fn.startswith(exp_prefix):
            langs = fn[len(exp_prefix):].split("_")
            if len(langs) != 2:
                continue
            lang1, lang2 = langs
            exp_name = exp_prefix + lang1 + '_' + lang2
            lang_pair = lang_lower[lang1] + "_sa-" + lang_lower[lang2] + "_sa"
            exps = []
            sub_fns = os.listdir(dump_path + fn)
            for sub_fn in sub_fns:
                if len(sub_fn) == 10 and sub_fn.isalnum():
                    exps.append(sub_fn)
            if len(exps) == 0:
                continue
            if len(exps) > 1:
                files = [os.path.join(dump_path + fn, f) for f in exps]
                files.sort(key=os.path.getmtime)
                exp_keys = [file.split('/')[-1] for file in files]
#                 exp_key = files[-1].split('/')[-1]
            else:
                exp_keys = exps
            is_cont = False
            for exp_key in exp_keys:
                exp_path = dump_path + exp_name + "/" + exp_key + '/'
                hypo_path  = exp_path + "hypotheses/"
                train_log_path = exp_path + 'train.log'
                ref_path = hypo_path + "ref." + lang_pair + ".test.txt"
                src_path = hypo_path + "src." + lang_pair + ".test.txt"
                with open(train_log_path) as infile:
                    lines = infile.readlines()
                results = []
                key = "__log__:"
                command_key = "--precont_topk"
                precont_topk = 0
                cont_lambda = 0
                max_tok_num = 0
                for line in lines:
                    if key in line:
                        result_str = line.strip().split(key)[1]
                        result = json.loads(result_str)
                        results.append(result)
                    elif command_key in line:
                        if precont_topk > 0:
                            continue
                        command_lines = line.split()
                        command_key_idx = command_lines.index(command_key)
                        precont_topk = int(command_lines[command_key_idx+1])
                        cont_lambda = float(command_lines[command_key_idx+3]
                                            .replace("\"", "").replace("\'", ""))
                        if command_key_idx + 4 < len(command_lines) - 1:
                            max_tok_num_key = command_lines[command_key_idx+4]
                            if max_tok_num_key == "--max_tok_num":
                                max_tok_num = int(command_lines[command_key_idx+5])
#                     elif "BLEU" in line:
#                         if precont_topk > 0:
#                             print(line)
#                 if precont_topk == 0:
#                     continue
                if len(results) == 0:
                    continue
                
                val_key_prefix = "valid_" + lang_pair
                test_key_prefix = "test_" + lang_pair
                rank_key_test = test_key_prefix + "_mt_bleu"
                rank_key_val = "valid_" + lang_pair + "_mt_bleu"
#                 for result in results:
#                     print(rank_key_val, result[rank_key_val])
                sorted_results = sorted(results, reverse=True, key=lambda x:x[rank_key_val])
                sorted_test_results = sorted(results, reverse=True, key=lambda x:x[rank_key_test])
                best_result = sorted_results[0]
                best_test_result = sorted_test_results[0]
        #         print(exp_name)
                key_bleu = test_key_prefix + "_mt_bleu"
                key_acc = test_key_prefix + "_mt_acc"
                key_ppl = test_key_prefix + "_mt_ppl"
                val_key_bleu = val_key_prefix + "_mt_bleu"
                val_key_acc = val_key_prefix + "_mt_acc"
                val_key_ppl = val_key_prefix + "_mt_ppl"
                epoch_id = best_result['epoch']
                test_epoch_id = best_test_result['epoch']
                result_csv_line = lang1 + '\t' + lang2 + '\t' + str(best_result[key_bleu]) + '\t' \
                     + str(best_result[key_ppl]) + '\t' + str(best_result[key_acc]) + '\t' \
                     + str(epoch_id) + '\t' + exp_name + '\t' + exp_key + '\t' 
                val_result_csv_line = lang1 + '\t' + lang2 + '\t' + str(best_result[val_key_bleu]) + '\t' \
                     + str(best_result[val_key_ppl]) + '\t' + str(best_result[val_key_acc]) + '\t' \
                     + str(epoch_id) + '\t' + exp_name + '\t' + exp_key + '\t' 

                best_result = best_test_result
                test_result_csv_line = lang1 + '\t' + lang2 + '\t' + str(best_result[key_bleu]) + '\t' \
                     + str(best_result[key_ppl]) + '\t' + str(best_result[key_acc]) + '\t' \
                     + str(test_epoch_id) + '\t' + exp_name + '\t' + exp_key + '\t'
                if precont_topk > 0:
                    is_cont = True
                if is_cont:
                    if precont_topk > 0 and max_tok_num == 0:
                        max_tok_num = 20
                    if print_result:
                        print("==============", lang1, lang2, "=============")
                        print("precont_topk", precont_topk, "cont_lambda", cont_lambda, "max_tok_num", max_tok_num)
                        # result_csv_line is the test result using best valid checkpoint
                        print(result_csv_line)
                        # test_result_csv_line is the best test result
                        print(test_result_csv_line)
                        # val_result_csv_line is the best valid result
                        print(val_result_csv_line)
                        print()

                if print_em:
                    if print_copy:
                        em_command = "python " + evaluator_path + "evaluator.py " + \
                                    "-ref " + ref_path + \
                                    "-pre " + src_path # pre_path  src_path
                    else:
                        pre_path = hypo_path + "hyp" + str(epoch_id) + "." + lang_pair + ".test_beam0.txt"
                        em_command = "python " + evaluator_path + "evaluator.py " + \
                                    "-ref " + ref_path + \
                                    "-pre " + pre_path # pre_path  src_path
                    print(em_command)

                if hypo_collection_path != None:
                    pre_path = hypo_path + "hyp" + str(epoch_id) + "." + lang_pair + ".test_beam0.txt"
                    hypo_suffix = ""
                    if is_cont:
                        hypo_suffix = "_" + str(precont_topk) + "_" + str(cont_lambda) + \
                                    "_" + str(max_tok_num) + "_" + str(best_result[key_bleu])
                        target_fn = "hyp" + str(epoch_id) + "." + lang_pair + ".test_beam0" + hypo_suffix + ".txt"
                        hypo_collection_path_full = hypo_collection_path + exp_name + "/"
                        if not os.path.exists(hypo_collection_path_full):
                            os.makedirs(hypo_collection_path_full)
                        copyfile(src_path, hypo_collection_path_full  + "src." + lang_pair + ".test.txt")
                        copyfile(pre_path, hypo_collection_path_full  + target_fn)
                        copyfile(ref_path, hypo_collection_path_full  + "ref." + lang_pair + ".test.txt")
    return

def read_results(exp_prefix, dump_path, 
                 print_result=True, print_em=False, print_copy=False, 
                 hypo_collection_path=None):
    beam_size = 5
    fns = os.listdir(dump_path)
    for fn in fns:
        if fn.startswith(exp_prefix):
            langs = fn[len(exp_prefix):].split("_")
            if len(langs) != 2:
                continue
            lang1, lang2 = langs
            
            exps = []
            sub_fns = os.listdir(dump_path + fn)
            for sub_fn in sub_fns:
                if len(sub_fn) == 10 and sub_fn.isalnum():
                    exps.append(sub_fn)
            if len(exps) == 0:
                continue
            if len(exps) > 1:
                files = [os.path.join(dump_path + fn, f) for f in exps]
                files.sort(key=os.path.getmtime)
                exp_key = files[-1].split('/')[-1]
            else:
                exp_key = exps[0]

            exp_name = exp_prefix + lang1 + '_' + lang2
            lang_pair = lang_lower[lang1] + "_sa-" + lang_lower[lang2] + "_sa"
            exp_path = dump_path + exp_name + "/" + exp_key + '/'
            hypo_path  = exp_path + "hypotheses/"
            train_log_path = exp_path + 'train.log'
            ref_path = hypo_path + "ref." + lang_pair + ".test.txt"
            src_path = hypo_path + "src." + lang_pair + ".test.txt"
            with open(train_log_path) as infile:
                lines = infile.readlines()
            results = []
            key = "__log__:"
            for line in lines:
                if key in line:
                    result_str = line.strip().split(key)[1]
                    result = json.loads(result_str)
                    results.append(result)
            if len(results) == 0:
                continue
            test_key_prefix = "test_" + lang_pair
            rank_key_test = test_key_prefix + "_mt_bleu"
            rank_key_val = "valid_" + lang_pair + "_mt_bleu"
            sorted_results = sorted(results, reverse=True, key=lambda x:x[rank_key_val])
            sorted_test_results = sorted(results, reverse=True, key=lambda x:x[rank_key_test])
            best_result = sorted_results[0]
            best_test_result = sorted_test_results[0]
    #         print(exp_name)
            key_bleu = test_key_prefix + "_mt_bleu"
            key_acc = test_key_prefix + "_mt_acc"
            key_ppl = test_key_prefix + "_mt_ppl"
            epoch_id = best_result['epoch']
            test_epoch_id = best_test_result['epoch']
            result_csv_line = lang1 + '\t' + lang2 + '\t' + str(best_result[key_bleu]) + '\t' \
                 + str(best_result[key_ppl]) + '\t' + str(best_result[key_acc]) + '\t' \
                 + str(epoch_id) + '\t' + exp_name + '\t' + exp_key + '\t' 
            
            best_result = best_test_result
            test_result_csv_line = lang1 + '\t' + lang2 + '\t' + str(best_result[key_bleu]) + '\t' \
                 + str(best_result[key_ppl]) + '\t' + str(best_result[key_acc]) + '\t' \
                 + str(test_epoch_id) + '\t' + exp_name + '\t' + exp_key + '\t'
            if print_result:
                print("==============", lang1, lang2, "=============")
                print(result_csv_line)
                print(test_result_csv_line)
                print()
            
            if print_em:
                print("==============", lang1, lang2, "=============")
                if print_copy:
                    em_command = "python " + evaluator_path + "evaluator.py " + \
                                "-ref " + ref_path + \
                                "-pre " + src_path # pre_path  src_path
                else:
                    pre_path = hypo_path + "hyp" + str(epoch_id) + "." + lang_pair + ".test_beam0.txt"
                    em_command = "python " + evaluator_path + "evaluator.py " + \
                                "-ref " + ref_path + \
                                "-pre " + pre_path # pre_path  src_path
                print(em_command)
            
            if hypo_collection_path != None:
                pre_path = hypo_path + "hyp" + str(epoch_id) + "." + lang_pair + ".test_beam0.txt"
                copyfile(src_path, hypo_collection_path  + "src." + lang_pair + ".test.txt")
                copyfile(pre_path, hypo_collection_path  + 
                         "hyp" + str(epoch_id) + "." + lang_pair + ".test_beam0.txt")
                copyfile(ref_path, hypo_collection_path  + 
                         "ref." + lang_pair + ".test.txt")
    return
    
    
def read_all_2_results(exp_prefix, dump_path,
                       print_result=True, print_em=False, print_copy=False, 
                       hypo_collection_path=None):
    langs = file_extensions.keys()
    fns = os.listdir(dump_path)
    for fn in fns:
        if fn.startswith(exp_prefix):
            lang2 = fn[len(exp_prefix):]
            print()
            print("++++++++++++++++++", fn , "++++++++++++++++")
            print()
            exps = []
            sub_fns = os.listdir(dump_path + fn)
            for sub_fn in sub_fns:
                if len(sub_fn) == 10 and sub_fn.isalnum():
                    exps.append(sub_fn)
            if len(exps) == 0:
                continue
            exp_keys = []
            if len(exps) > 1:
                files = [os.path.join(dump_path + fn, f) for f in exps]
                files.sort(key=os.path.getmtime)
                exp_keys = [file.split('/')[-1] for file in files]
                exp_key = exp_keys[-1]
            else:
                exp_keys = exps

    #         exp_key = exp_keys[-1]
            exp_name = exp_prefix + lang2
            exp_keys = exp_keys[-1:]
            for exp_key in exp_keys:
                exp_path = dump_path + exp_name + "/" + exp_key + '/'
                checkpoint_fns = os.listdir(exp_path)
                for fn in checkpoint_fns:
                    if fn.startswith("best-valid_"):
                        checkpoint_path = exp_path + fn #"best-valid_" + lang_pair + "_mt_bleu.pth"
                        lang_pair = fn[len("best-valid_"):-len("_mt_bleu.pth")]
                        lang12 = lang_pair.split("-")
                        lang1 = lang_upper[lang12[0][:-len("-sa")]]
                        lang2 = lang_upper[lang12[1][:-len("-sa")]]
                        
                        hypo_path  = exp_path + "hypotheses/"
                        train_log_path = exp_path + 'train.log'
                        ref_path = hypo_path + "ref." + lang_pair + ".test.txt"
                        src_path = hypo_path + "src." + lang_pair + ".test.txt"
                        with open(train_log_path) as infile:
                            lines = infile.readlines()
                        results = []
                        key = "__log__:"
                        for line in lines:
                            if key in line:
                                result_str = line.strip().split(key)[1]
                                result = json.loads(result_str)
                                results.append(result)
                        if len(results) == 0:
                            continue
                        test_key_prefix = "test_" + lang_pair
                        rank_key_test = test_key_prefix + "_mt_bleu"
                        rank_key_val = "valid_" + lang_pair + "_mt_bleu"
                        sorted_results = sorted(results, reverse=True, key=lambda x:x[rank_key_val])
                        sorted_test_results = sorted(results, reverse=True, key=lambda x:x[rank_key_test])
                        best_result = sorted_results[0]
                        best_test_result = sorted_test_results[0]
                #         print(exp_name)
                        key_bleu = test_key_prefix + "_mt_bleu"
                        key_acc = test_key_prefix + "_mt_acc"
                        key_ppl = test_key_prefix + "_mt_ppl"
                        epoch_id = best_result['epoch']
                        test_epoch_id = best_test_result['epoch']
                        result_csv_line = lang1 + '\t' + lang2 + '\t' + str(best_result[key_bleu]) + '\t' \
                             + str(best_result[key_ppl]) + '\t' + str(best_result[key_acc]) + '\t' \
                             + str(epoch_id) + '\t' + exp_name + '\t' + exp_key + '\t' 
                        
                        best_result = best_test_result
                        test_result_csv_line = lang1 + '\t' + lang2 + '\t' + str(best_result[key_bleu]) + '\t' \
                             + str(best_result[key_ppl]) + '\t' + str(best_result[key_acc]) + '\t' \
                             + str(test_epoch_id) + '\t' + exp_name + '\t' + exp_key + '\t' 
                        if print_result:
                            print("==============", lang1, lang2, "=============")
                            print(result_csv_line)
                            print(test_result_csv_line)
                            print()
    
                        if print_em:
                            if print_copy:
                                em_command = "python " + evaluator_path + "evaluator.py " + \
                                            "-ref " + ref_path + \
                                            "-pre " + src_path # pre_path  src_path
                            else:
                                pre_path = hypo_path + "hyp" + str(epoch_id) + "." + lang_pair + ".test_beam0.txt"
                                em_command = "python " + evaluator_path + "evaluator.py " + \
                                            "-ref " + ref_path + \
                                            "-pre " + pre_path # pre_path  src_path
                            print(em_command)
                        if hypo_collection_path != None:
                            copyfile(src_path, hypo_collection_path  + "src." + lang_pair + ".test.txt")
                            copyfile(pre_path, hypo_collection_path  + 
                                     "hyp" + str(epoch_id) + "." + lang_pair + ".test_beam0.txt")
                            copyfile(ref_path, hypo_collection_path  + 
                                     "ref." + lang_pair + ".test.txt")
    return

def get_multilingual_train_commands(langs, model_path, data_path_prefix, exp_prefix, model_exp_prefix,
                       max_epoch=200, max_len=400, beam_size=10,
                       is_reloaded=True, is_dobf=True, is_transferred=False, is_print=True,
                       is_ae=True):
    for lang2 in langs:
        exp_name = exp_prefix + lang2
        mt_steps_list = []
        validation_metrics_list = []
        lgs_list = []
        lang2_sa = lang_lower[lang2] + "_sa"
        exp_path_prefix = dump_path + exp_name + "/"
        data_path = data_path_prefix + lang2 + "/"
        if is_print:
            print("==============", lang2, "=============")
        for lang1 in langs:
            if lang2 == lang1:
                continue
            lang_pair = lang_lower[lang1] + "_sa-" + lang_lower[lang2] + "_sa"
            mt_steps_list.append(lang_pair)
            val_metrics = "valid_" + lang_pair + "_mt_bleu"
            validation_metrics_list.append(val_metrics)
            lgs_list.append(lang_lower[lang1] + "_sa")
        lgs_list.append(lang2_sa)   
        mt_steps = ",".join(mt_steps_list) 
        ae_steps = lang2_sa
        lgs = "-".join(lgs_list) 
        validation_metrics = ",".join(validation_metrics_list) 
        train_command = "python train.py " + \
        "--exp_name " + exp_name + " " + \
        "--dump_path /home/mingzhu/CodeModel/CodeGen/dumppath1 " +  \
        "--data_path " + data_path + " " + \
        "--mt_steps " + mt_steps + " " + \
        "--encoder_only False " + \
        "--n_layers 0  " + \
        "--lgs " + lgs  + " " + \
        "--max_vocab 64000 " + \
        "--gelu_activation true " + \
        "--roberta_mode false   " + \
        "--amp 2  " + \
        "--fp16 true  " + \
        "--tokens_per_batch 3000  " + \
        "--group_by_size true " + \
        "--max_batch_size 128  " +  \
        "--epoch_size 10000  " +   \
        "--split_data_accross_gpu global  " +  \
        "--has_sentences_ids true  " + \
        "--optimizer 'adam_inverse_sqrt,warmup_updates=10000,lr=0.0001,weight_decay=0.01'  " + \
        "--eval_bleu true   " + \
        "--eval_computation false   " + \
        "--generate_hypothesis true   " + \
        "--validation_metrics " +  validation_metrics + " " + \
        "--eval_only false" + " " \
        "--max_epoch " + str(max_epoch) + " " +  \
        "--beam_size " + str(beam_size) + " " + \
        "--max_len " + str(max_len) + " "
        if is_ae:
            train_command += "--ae_steps " + ae_steps  + " " + \
                             "--lambda_ae " + "0:1,30000:0.1,100000:0" + " "
        if is_dobf:
            train_command += "--n_layers_encoder 12   " + \
                                "--n_layers_decoder 6  " + \
                                "--emb_dim 768   " + \
                                "--n_heads 12   "
        else:
            train_command += "--n_layers_encoder 6   " + \
                                "--n_layers_decoder 6  " + \
                                "--emb_dim 1024   "

        if is_reloaded:
            if is_transferred:
                # Continue training on another checkpoint
                model_exp_name = model_exp_prefix + lang2
                model_exp_path_prefix = dump_path + model_exp_name + "/"
                exps = []
                sub_fns = os.listdir(model_exp_path_prefix)
                for sub_fn in sub_fns:
                    if len(sub_fn) == 10 and sub_fn.isalnum():
                        exps.append(sub_fn)
                if len(exps) > 1:
                    files = [os.path.join(model_exp_path_prefix, f) for f in exps]
                    files.sort(key=os.path.getmtime)
                    exp_key = files[-1].split('/')[-1]
                else:
                    exp_key = exps[0]
                model_exp_path = model_exp_path_prefix + exp_key + '/'
                for lang_pair in mt_steps_list:
                    model_path = model_exp_path + "best-valid_" + lang_pair + "_mt_bleu.pth"
                    if not os.path.exists(model_path):
                        print(model_path + " not exist")
                        continue
                    train_command += "--reload_model " + model_path + ',' + model_path + " "
                    print(train_command)
            else:
                train_command += "--reload_model " + model_path + ',' + model_path + " "
                print(train_command)
                
                
def check_exp_path(model_exp_path_prefix, lang_pair):
    exps = []
    sub_fns = os.listdir(model_exp_path_prefix)
    for sub_fn in sub_fns:
        if len(sub_fn) == 10 and sub_fn.isalnum():
            exps.append(sub_fn)
    if len(exps) > 1:
        exp_paths = [os.path.join(model_exp_path_prefix, f) for f in exps]
        valid_exp_paths = []
        for exp_pth in exp_paths:
            # best-valid_cpp_sa-java_sa_mt_bleu.pth
            check_pt = exp_pth + "/" + "best-valid_" + lang_pair + "_mt_bleu.pth"
            if os.path.exists(check_pt):
                valid_exp_paths.append(exp_pth)
        valid_exp_paths.sort(key=os.path.getmtime)
        exp_key = valid_exp_paths[-1].split('/')[-1]
    else:
        exp_key = exps[0]
    return exp_key

def get_train_commands(langs, model_path, data_path_prefix, exp_prefix, model_exp_prefix,
                       max_epoch=200, max_len=400, beam_size=10,
                       is_reloaded=True, is_dobf=True, is_transferred=False, is_print=True, 
                       is_cont=False, precont_topk=0, cont_lambda=0, max_tok_num=20):
    for lang1 in langs:
        for lang2 in langs:
            if lang2 == lang1:
                continue
            if is_print:
                print("==============", lang1, lang2, "=============")
            exp_name = exp_prefix + lang1 + '_' + lang2
            data_path = data_path_prefix + lang1 + '-' + lang2 + "/"
            if not os.path.exists(data_path):
                data_path = data_path_prefix + lang2 + '-' + lang1 + "/"

            lang_pair = lang_lower[lang1] + "_sa-" + lang_lower[lang2] + "_sa"
            mt_steps = lang_pair
            validation_metrics = "valid_" + lang_pair + "_mt_bleu"
            lgs = lang_pair
            exp_path_prefix = dump_path + exp_name + "/"

            train_command = "python train.py " + \
            "--exp_name " + exp_name + " " + \
            "--dump_path /home/mingzhu/CodeModel/CodeGen/dumppath1 " +  \
            "--data_path " + data_path + " " + \
            "--mt_steps " + mt_steps + " " + \
            "--encoder_only False " + \
            "--n_layers 0  " + \
            "--lgs " + lgs  + " " + \
            "--max_vocab 64000 " + \
            "--gelu_activation true " + \
            "--roberta_mode false   " + \
            "--amp 2  " + \
            "--fp16 true  " + \
            "--tokens_per_batch 3000  " + \
            "--group_by_size true " + \
            "--max_batch_size 128  " +  \
            "--epoch_size 10000  " +   \
            "--split_data_accross_gpu global  " +  \
            "--has_sentences_ids true  " + \
            "--optimizer 'adam_inverse_sqrt,warmup_updates=10000,lr=0.0001,weight_decay=0.01'  " + \
            "--eval_bleu true   " + \
            "--eval_computation false   " + \
            "--generate_hypothesis true   " + \
            "--validation_metrics " +  validation_metrics + " " + \
            "--eval_only false" + " " \
            "--max_epoch " + str(max_epoch) + " " +  \
            "--beam_size " + str(beam_size) + " " + \
            "--max_len " + str(max_len) + " "
            if is_dobf:
                train_command += "--n_layers_encoder 12   " + \
                                    "--n_layers_decoder 6  " + \
                                    "--emb_dim 768   " + \
                                    "--n_heads 12   "
            else:
                train_command += "--n_layers_encoder 6   " + \
                                    "--n_layers_decoder 6  " + \
                                    "--emb_dim 1024   "
            if is_reloaded:
                if is_transferred:
                    # Continue training on another checkpoint

                    model_exp_name = model_exp_prefix + lang1 + '_' + lang2
                    model_exp_path_prefix = dump_path + model_exp_name + "/"
                    if not os.path.exists(model_exp_path_prefix):
                        continue
                    exp_key = check_exp_path(model_exp_path_prefix, lang_pair)
                    model_exp_path = model_exp_path_prefix + exp_key + '/'
                    
                    model_path = model_exp_path + "best-valid_" + lang_pair + "_mt_bleu.pth"
                    if not os.path.exists(model_path):
                        print(model_path)
                        continue
                train_command += "--reload_model " + model_path + ',' + model_path + " "
                if is_cont:
                    train_command += "--eval_only True --conditional_generation True  --precont_topk " + \
                                        str(precont_topk) + " --cont_lambda " + str(cont_lambda) +\
                                        " --max_tok_num " + str(max_tok_num)

            print(train_command)
    return

