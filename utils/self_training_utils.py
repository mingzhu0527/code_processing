from sample_utils import *
from inference_utils import *
from leetcode_exec_utils import *
from common_utils import *

def get_preds_lang_dict_codex(merged_filtered_codex_dict, call_dict_gold, 
                                  model_type="codex", data_name="xlcost", tag="test", label=""):
    preds_lang_dict_path = cached_path + \
        model_type + "_" + tag + "_" + data_name + "_preds_lang_dict" + label + ".pkl"
    preds_pid_dict_path = cached_path + \
        model_type + "_" + tag + "_" + data_name + "_preds_pid_dict" + label + ".pkl"
    if os.path.exists(preds_lang_dict_path) and os.path.exists(preds_pid_dict_path):
        with open(preds_lang_dict_path, 'rb') as infile:
            preds_lang_dict = pickle.load(infile)
        with open(preds_pid_dict_path, 'rb') as infile:
            preds_pid_dict = pickle.load(infile)
        return preds_lang_dict, preds_pid_dict
    new_codex_dict = {}
    new_pids_dict = {}
    for lang1, lang2 in merged_filtered_codex_dict.keys():
        if lang1 not in new_langs or lang2 not in new_langs:
            continue
        new_codex_dict[(lang1, lang2)] = []
        new_pids_dict[(lang1, lang2)] = []
        (map_dict, reverse_map_dict), pids, programs_gold, \
              (result_id_dict_gold, result_key_dict_gold, error_type_dict_gold) = call_dict_gold[(lang1, lang2)]
        codex_dic = merged_filtered_codex_dict[(lang1, lang2)]
        for pid in pids:
            if pid in codex_dic:
                new_codex_dict[(lang1, lang2)].append([codex_dic[pid]["output"]])
                new_pids_dict[(lang1, lang2)].append(pid)
#             else:
#                 new_codex_dict[(lang1, lang2)].append([""])
    with open(preds_lang_dict_path, 'wb') as outfile:
        pickle.dump(new_codex_dict, outfile)
    with open(preds_pid_dict_path, 'wb') as outfile:
        pickle.dump(new_pids_dict, outfile)
    return new_codex_dict, new_pids_dict

def get_preds_lang_dict(lang_pairs, model_type, device, src_codes, tgt_codes, function_data_path, 
                        sample_size=5, temperature=0.5,
                        data_name="xlcost", tag='test', exp_suffix="_translation_exec_function/", label=""):
    preds_lang_dict_path = cached_path + model_type + "_" + tag + "_" + data_name + "_preds_lang_dict" + label + ".pkl"
    if os.path.exists(preds_lang_dict_path):
        with open(preds_lang_dict_path, 'rb') as infile:
            preds_lang_dict = pickle.load(infile)
            return preds_lang_dict
    preds_lang_dict = {}
    for lang1, lang2 in lang_pairs:
        print(lang1, lang2)
        eval_examples, eval_features, eval_dataloader, model, tokenizer, args, decoder_sid = inference_prepro(
            lang1, lang2, model_type, device, src_codes, tgt_codes, function_data_path, tag, exp_suffix)
        is_eval = tgt_codes==[]
        preds, eval_result = generation_multiple(eval_examples, eval_dataloader, 
                                                     model, tokenizer, args, device, 
                                                     decoder_sid, is_eval, sample_size, temperature)
        preds_lang_dict[(lang1, lang2)] = preds
    with open(preds_lang_dict_path, 'wb') as outfile:
        pickle.dump(preds_lang_dict, outfile)
    return preds_lang_dict

def get_preds_lang_dict_new(lang_pairs, model_type, device, src_codes, tgt_codes, function_data_path, 
                        sample_size=5, temperature=0.5,
                        data_name="xlcost", tag='test', exp_suffix="_translation_exec_function/", label="", do_sample=True):
    preds_lang_dict_path = cached_path + model_type + "_" + tag + "_" + data_name + "_preds_lang_dict" + label + ".pkl"
    if os.path.exists(preds_lang_dict_path):
        with open(preds_lang_dict_path, 'rb') as infile:
            preds_lang_dict = pickle.load(infile)
            return preds_lang_dict
    preds_lang_dict = {}
    for lang1, lang2 in lang_pairs:
        print(lang1, lang2)
        eval_examples, eval_features, eval_dataloader, model, tokenizer, args, decoder_sid = inference_prepro(
            lang1, lang2, model_type, device, src_codes, tgt_codes, function_data_path, tag, exp_suffix)
        is_eval = tgt_codes==[]
        if sample_size == 1:
            preds, eval_result = generation_single(eval_examples, eval_dataloader, 
                                            model, tokenizer, args, device, 
                                            do_sample, temperature)
            preds = [[x] for x in preds]
        else:
            preds, eval_result = generation_multiple(eval_examples, eval_dataloader, 
                                                         model, tokenizer, args, device, 
                                                         decoder_sid, is_eval, sample_size, temperature)
        preds_lang_dict[(lang1, lang2)] = preds
    with open(preds_lang_dict_path, 'wb') as outfile:
        pickle.dump(preds_lang_dict, outfile)
    return preds_lang_dict


def get_repeated_sampling_dict(lang_pairs, model_type, device, 
                        src_codes, tgt_codes, function_data_path, function_id_lang_dic, 
                        data_name="xlcost", tag='test', exp_suffix="_translation_exec_function/",
                        is_eval=True, num_iterations=1, num_samples=5, temperature=0.5):
    rsd_path = cached_path + model_type + "_" + tag + "_" + data_name + "_repeated_sampling_dict.pkl"
    if os.path.exists(rsd_path):
        with open(rsd_path, 'rb') as infile:
            repeated_sampling_dict = pickle.load(infile)
            return repeated_sampling_dict
    call_dict_gold = get_call_dict_gold(lang_pairs, function_data_path, function_id_lang_dic, data_name, tag)
    repeated_sampling_dict = {}
    for lang1, lang2 in lang_pairs:
        print(lang1, lang2)
        eval_examples, eval_features, eval_dataloader, model, tokenizer, args, decoder_sid = inference_prepro(
            lang1, lang2, model_type, device, src_codes, tgt_codes, function_data_path, tag, exp_suffix)
        
        (_, reverse_map_dict), pids, _, (result_id_dict_gold, _, _) = call_dict_gold[(lang1, lang2)]
        buggy_pids, failed_test_pids, good_hypo_dict = repeated_sampling(
                         pids, eval_examples, eval_features, 
                         function_id_lang_dic, reverse_map_dict, result_id_dict_gold, lang2, model_type,
                         model, tokenizer, args, device, decoder_sid, is_eval, 
                         num_iterations, num_samples, temperature)
        repeated_sampling_dict[(lang1, lang2)] = [buggy_pids, failed_test_pids, good_hypo_dict]
    with open(rsd_path, 'wb') as outfile:
        pickle.dump(repeated_sampling_dict, outfile)
    return repeated_sampling_dict

def repeated_sampling(pids, eval_examples, eval_features, 
                     function_id_lang_dic, reverse_map_dict, result_id_dict_gold, lang,
                     model_type, model, tokenizer, args, device, decoder_sid, is_eval=True,
                     num_iterations=1, num_samples=5, temperature=0.5):
    failed_pids = pids
    good_hypo_dict = {}
    # how to add randomness to the sampling?
    for i in range(num_iterations):
        buggy_pids, failed_test_pids, passed_hypo_dict = sampling_filtering(
                                             failed_pids, eval_examples, eval_features, 
                                             function_id_lang_dic, reverse_map_dict, result_id_dict_gold, lang,
                                             model_type, model, tokenizer, args, device, decoder_sid, is_eval,
                                             num_samples, temperature)
        temperature = 0.9
        for pid, hypo_list in passed_hypo_dict.items():
            good_hypo_dict[pid] = hypo_list
        failed_pids = buggy_pids + failed_test_pids
        if len(failed_pids) == 0:
            break
        torch.cuda.empty_cache()
    return buggy_pids, failed_test_pids, good_hypo_dict

def sampling_filtering(pids, eval_examples, eval_features, function_id_lang_dic,
                       reverse_map_dict, result_id_dict_gold, lang, model_type, model, tokenizer, args, device,
                       decoder_sid=None, is_eval=True, num_samples=5, temperature=0.5):
    selected_eval_examples, selected_eval_dataloader = get_eval_data_by_pid(eval_examples, eval_features, 
                                                               pids, reverse_map_dict, args.eval_batch_size)
    torch.cuda.empty_cache()
    preds, eval_result = generation_multiple(selected_eval_examples, 
                                             selected_eval_dataloader, 
                                             model, tokenizer, args, device, 
                                             decoder_sid, is_eval, num_samples, temperature)
    programs, program_id_dict, program_dict = prep_exec_hypo(preds, pids,
                                                             function_id_lang_dic, lang, model_type)
    lang_results = p_map(file_executors[lang], programs)
    result_id_dict, result_key_dict, error_type_dict = result_mapping(lang_results, program_id_dict, 
                                                                      pids, lang)
    buggy_pids, failed_test_pids, passed_hypo_dict = hypo_filtering(result_id_dict, 
                                                                    result_id_dict_gold, 
                                                                    result_key_dict, 
                                                                    program_dict)
    print(len(buggy_pids),len(failed_test_pids), len(passed_hypo_dict), len(pids))
    return buggy_pids, failed_test_pids, passed_hypo_dict