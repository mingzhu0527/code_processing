from run import *
from bleu import compute_bleu, _bleu
from common_utils import *
import torch

def print_exec_results(exec_rate_dict, new_langs=new_langs):
    for lang1 in new_langs:
        result_line = ""
        for lang2 in new_langs:
            if lang2 == lang1:
                result_line += "\\t"
                continue
            if (lang1, lang2) not in exec_rate_dict:
                result_line += "\\t"
                continue
            result_line += str(round(exec_rate_dict[(lang1, lang2)]*100, 2)) + "\\t"
        print(result_line)
    return 

def get_exec_rate_dict(hypo_filtering_dict):
    exec_rate_dict = {}
    for lang1 in langs:
        for lang2 in langs:
            if (lang1, lang2) in hypo_filtering_dict:
    #             _, pids, _, (result_id_dict_gold, _, _) = call_dict_gold[(lang1, lang2)]
                buggy_pids, failed_test_pids, passed_hypo_dict = hypo_filtering_dict[(lang1, lang2)]
                exec_rate = len(passed_hypo_dict)/(len(buggy_pids)
                                                   + len(failed_test_pids) + len(passed_hypo_dict))
                exec_rate_dict[(lang1, lang2)] = exec_rate
    return exec_rate_dict

def hypo_filtering_pids(result_id_dict, result_id_dict_gold, result_key_dict, program_dict, pids):
    non_buggy_pid_dict = {}
    buggy_pids = []
    pids_set = set(pids)
    for pid, result_keys in result_key_dict.items():
        if pid not in pids_set:
            continue
        good_samples = []
        for i, result_key in enumerate(result_keys):
            if result_key != 'error':
                if program_dict[pid][i] != "":
                    good_samples.append(i)
        if len(good_samples) == 0:
            buggy_pids.append(pid)
        else:
            non_buggy_pid_dict[pid] = good_samples
            
    failed_test_pids = []
    passed_test_pid_dict = {}
    for pid, idx_list in non_buggy_pid_dict.items():
        pid_results = result_id_dict[pid]
        pid_result_gold = result_id_dict_gold[pid][0]
        pid_results_filtered = []
        for idx in idx_list:
            pid_result = pid_results[idx]
            if pid_result_gold != "" and pid_result == pid_result_gold:
                pid_results_filtered.append(idx)
        if len(pid_results_filtered) == 0:
            failed_test_pids.append(pid)
        else:
            passed_test_pid_dict[pid] = pid_results_filtered
            
    passed_hypo_dict = {}
    for pid, idx_list in passed_test_pid_dict.items():
        hypos = [program_dict[pid][x] for x in idx_list]
        passed_hypo_dict[pid] = hypos
    return buggy_pids, failed_test_pids, passed_hypo_dict

def get_hypo_filtering_dict_pids(lang_pairs, call_dict_gold, call_dict_hypo, pids_dict):
    hypo_filtering_dict = {}
    for lang1, lang2 in lang_pairs:
        print(lang1, lang2)
        pids = pids_dict[(lang1, lang2)]
        _, _, _, (result_id_dict_gold, _, _) = call_dict_gold[(lang1, lang2)]
        _, _, _, program_dict, (result_id_dict, result_key_dict, _) = call_dict_hypo[(lang1, lang2)]
        buggy_pids, failed_test_pids, passed_hypo_dict = hypo_filtering_pids(result_id_dict, 
                                                                        result_id_dict_gold, 
                                                                        result_key_dict, 
                                                                        program_dict, pids)
        hypo_filtering_dict[(lang1, lang2)] = [buggy_pids, failed_test_pids, passed_hypo_dict]
        print(len(buggy_pids),len(failed_test_pids), len(passed_hypo_dict), len(pids))
    return hypo_filtering_dict

def hypo_filtering(result_id_dict, result_id_dict_gold, result_key_dict, program_dict):
    non_buggy_pid_dict = {}
    buggy_pids = []
    for pid, result_keys in result_key_dict.items():
        good_samples = []
        for i, result_key in enumerate(result_keys):
            if result_key != 'error':
                if program_dict[pid][i] != "":
                    good_samples.append(i)
        if len(good_samples) == 0:
            buggy_pids.append(pid)
        else:
            non_buggy_pid_dict[pid] = good_samples
            
    failed_test_pids = []
    passed_test_pid_dict = {}
    for pid, idx_list in non_buggy_pid_dict.items():
        pid_results = result_id_dict[pid]
        pid_result_gold = result_id_dict_gold[pid][0]
        pid_results_filtered = []
        for idx in idx_list:
            pid_result = pid_results[idx]
            if pid_result_gold != "" and pid_result == pid_result_gold:
                pid_results_filtered.append(idx)
        if len(pid_results_filtered) == 0:
            failed_test_pids.append(pid)
        else:
            passed_test_pid_dict[pid] = pid_results_filtered
            
    passed_hypo_dict = {}
    for pid, idx_list in passed_test_pid_dict.items():
        hypos = [program_dict[pid][x] for x in idx_list]
        passed_hypo_dict[pid] = hypos
    return buggy_pids, failed_test_pids, passed_hypo_dict

def get_hypo_filtering_dict(lang_pairs, call_dict_gold, call_dict_hypo):
    hypo_filtering_dict = {}
    for lang1, lang2 in lang_pairs:
        print(lang1, lang2)
        _, pids, _, (result_id_dict_gold, _, _) = call_dict_gold[(lang1, lang2)]
        _, _, _, program_dict, (result_id_dict, result_key_dict, _) = call_dict_hypo[(lang1, lang2)]
        buggy_pids, failed_test_pids, passed_hypo_dict = hypo_filtering(result_id_dict, 
                                                                        result_id_dict_gold, 
                                                                        result_key_dict, 
                                                                        program_dict)
        hypo_filtering_dict[(lang1, lang2)] = [buggy_pids, failed_test_pids, passed_hypo_dict]
        print(len(buggy_pids),len(failed_test_pids), len(passed_hypo_dict), len(pids))
    return hypo_filtering_dict


def get_out_fn(do_sample, temperature, beam_size):
    out_fn = "output." 
    if do_sample:
        out_fn += str(int(temperature*100)) + ".sample"
    else:
        out_fn += str(beam_size) + ".beam"
    return out_fn

def generation_multiple(eval_examples, eval_dataloader, model, tokenizer, args, device, 
                      decoder_sid=None, is_eval=True, sample_size=5, temperature=0.5):
    out_fn = get_out_fn(True, temperature, args.beam_size)
    preds = get_sample_generations(eval_dataloader, model, tokenizer, args, device, 
                                   decoder_sid, sample_size, temperature)
    eval_result = None
    if is_eval:
        accs, dev_bleus, dev_bleu, xmatch = eval_bleu_samples(eval_examples, args, preds, out_fn)
        eval_result = (accs, dev_bleus, dev_bleu, xmatch)
    return preds, eval_result


def generation_single(eval_examples, eval_dataloader, model, tokenizer, args, device, 
                      do_sample, decoder_sid=None, is_eval=True, temperature=0.5):
    out_fn = get_out_fn(do_sample, temperature, args.beam_size)
    preds = get_generation_single(eval_dataloader, model, tokenizer, args, device, 
                                  do_sample, decoder_sid, temperature)
    eval_result = None
    if is_eval:
        dev_bleu, xmatch = eval_bleu(eval_examples, args, preds, out_fn)
        eval_result = (dev_bleu, xmatch)
    return preds, eval_result

def get_generation(batch, model, tokenizer, args, beam_size, do_sample, decoder_sid=None, temperature=0.5):
    top_preds = []
    source_ids,source_mask= batch  
    with torch.no_grad():
        if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)
        else:
            if decoder_sid:
                preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=beam_size,
                                       do_sample=do_sample,
                                       temperature=temperature,
                                       early_stopping=False, # 如果是summarize就设为True
                                       max_length=args.max_target_length,
                                       decoder_start_token_id=tokenizer.sep_token_id)
            else:
                preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=beam_size,
                                       do_sample=do_sample,
                                       temperature=temperature,
                                       early_stopping=False, # 如果是summarize就设为True
                                       max_length=args.max_target_length)
        top_preds = list(preds.cpu().numpy())
    return top_preds

def get_generation_single(eval_dataloader, model, tokenizer, args, 
                      device, do_sample, decoder_sid=None, temperature=0.5):
    pred_ids = []
    beam_size = args.beam_size
    if do_sample:
        beam_size = 1
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids,source_mask= batch  
        top_preds = get_generation(batch, model, tokenizer, args, beam_size, do_sample, decoder_sid, temperature)
        pred_ids.extend(top_preds)
    hypos = [tokenizer.decode(pred, skip_special_tokens=True, 
                              clean_up_tokenization_spaces=False)
                              for pred in pred_ids]
    return hypos

def get_generation_demo(batch, model, tokenizer, args, 
                      device, do_sample, decoder_sid=None, temperature=0.5):
    
    pred_ids = []
    beam_size = args.beam_size
    if do_sample:
        beam_size = 1
    batch = tuple(t.to(device) for t in batch)
    top_preds = get_generation(batch, model, tokenizer, args, beam_size, do_sample, decoder_sid, temperature)
    pred_ids = top_preds
    hypos = [tokenizer.decode(pred, skip_special_tokens=True, 
                              clean_up_tokenization_spaces=False)
                              for pred in pred_ids]
    return hypos

def eval_bleu(eval_examples, args, hypos, out_fn):
    predictions=[]
    accs=[]
    idx = 0
    ref_path = os.path.join(args.output_dir, "test_{}.gold".format(str(idx)))
    hypo_path = os.path.join(args.output_dir, out_fn)
    with open(hypo_path,'w') as hypo_f, open(ref_path,'w') as ref_f:
        for hypo,gold in zip(hypos, eval_examples):
            predictions.append(str(gold.idx)+'\t'+hypo)
            hypo_f.write(hypo+'\n')
            ref_f.write(gold.target+'\n')    
            accs.append(hypo==gold.target)
    
    dev_bleu_raw = _bleu(ref_path, hypo_path)
    dev_bleu=round(dev_bleu_raw, 2)
    xmatch = str(round(np.mean(accs)*100,4))
    print("  %s = %s "%("bleu-4",str(dev_bleu)))
    print("  %s = %s "%("xMatch", xmatch))
    print("  "+"*"*20)  
    return dev_bleu, xmatch
    
# https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/generation_utils.py#1196
# 6. determine generation mode
# is_constraint_gen_mode = constraints is not None or force_words_ids is not None
# is_greedy_gen_mode = (
#     (num_beams == 1) and (num_beam_groups == 1) and do_sample is False and not is_constraint_gen_mode
# )
# is_sample_gen_mode = (
#     (num_beams == 1) and (num_beam_groups == 1) and do_sample is True and not is_constraint_gen_mode
# )
# is_beam_gen_mode = (
#     (num_beams > 1) and (num_beam_groups == 1) and do_sample is False and not is_constraint_gen_mode
# )
# is_beam_sample_gen_mode = (
#     (num_beams > 1) and (num_beam_groups == 1) and do_sample is True and not is_constraint_gen_mode
# )
# is_group_beam_gen_mode = (num_beams > 1) and (num_beam_groups > 1) and not is_constraint_gen_mode
def get_sample_generations(eval_dataloader, model, tokenizer, args, 
                      device, decoder_sid=None, sample_size=1, temperature=0.5):
    hypos=[]
    pred_ids = []
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids,source_mask= batch                  
        with torch.no_grad():
            if args.model_type == 'roberta':
                    preds = model(source_ids=source_ids, source_mask=source_mask)
            else:
#  https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/generation_utils.py#L831
#  See all the configurable parameters
                if decoder_sid:
                    preds = model.generate(source_ids,
                                           attention_mask=source_mask,
                                           use_cache=True,
                                           num_beams=1,
                                           do_sample=True,
                                           temperature=temperature,
                                           early_stopping=False, # 如果是summarize就设为True
                                           max_length=args.max_target_length,
                                           num_return_sequences=sample_size,
                                           decoder_start_token_id=decoder_sid)
                else:
                    preds = model.generate(source_ids,
                                           attention_mask=source_mask,
                                           use_cache=True,
                                           num_beams=1,
                                           do_sample=True,
                                           temperature=temperature,
                                           early_stopping=False, # 如果是summarize就设为True
                                           max_length=args.max_target_length,
                                           num_return_sequences=sample_size)
                preds_reshape = preds.reshape(source_ids.size()[0], sample_size, -1)
                top_preds = list(preds_reshape.cpu().numpy())
                pred_ids.extend(top_preds)
    for samples in tqdm(pred_ids):
        samples_dec = tokenizer.batch_decode(samples, skip_special_tokens=True, 
                              clean_up_tokenization_spaces=False)
#         for sample in samples:
#             sample_dec = tokenizer.decode(sample, skip_special_tokens=True, 
#                               clean_up_tokenization_spaces=False)
#             samples_dec.append(sample_dec)
        hypos.append(samples_dec)
    return hypos

def eval_bleu_samples(eval_examples, args, hypos, out_fn):
    idx = 0
    gold_fn = os.path.join(args.output_dir,"test_{}.gold".format(str(idx)))
    with open(gold_fn,'w') as outfile:
        for gold in eval_examples:
            outfile.write(gold.target+'\n') 
    accs=[]
    dev_bleus = []
    for index, (samples, gold) in enumerate(zip(hypos, eval_examples)):
        acc = False
        sample_bleus = []
        sample_fn = os.path.join(args.output_dir, out_fn + "." + str(index))
        with open(sample_fn,'w') as outfile:
            for sample in samples:
                outfile.write(sample + '\n')
                if sample==gold.target:
                    acc = True
                sample_bleu = round(compute_bleu(gold.target, sample, 4, True)[0], 2)
        sample_bleus.append(sample_bleu)
        dev_bleus.append(sample_bleus)
        accs.append(acc)
    dev_bleu = str(round(np.mean([np.max(x) for x in dev_bleus])*100,4))
    xmatch = str(round(np.mean(accs)*100,4))
    print("  %s = %s "%("bleu-4", dev_bleu))
    print("  %s = %s "%("xMatch", xmatch))
    print("  "+"*"*20)  
    return dev_bleus, accs, dev_bleu, xmatch

def compile_samples(eval_batch_size, beam_size, do_sample, temperature, sample_size=1):
    eval_examples, eval_dataloader = get_eval_dataloader(test_file, eval_batch_size)
    out_fn = "output." 
    if do_sample:
        out_fn += str(temperature*100) + ".sample"
    else:
        out_fn += str(beam_size) + ".beam"
    if sample_size > 1:
        out_fn += "_" + str(sample_size)
        sample_generation(eval_examples, eval_dataloader, model, args, 
                          beam_size, do_sample, temperature, out_fn, sample_size)
    else:
        sample_generation_single(eval_examples, eval_dataloader, model, args, 
                          beam_size, do_sample, temperature, out_fn)
    # Free gpu memory after training
    torch.cuda.empty_cache()
    
    
    # detok 参考Utils - code viewer和comps
    if sample_size == 1:
        hypo_file = output_dir + out_fn 
        new_fn = hypo_file + '.json'
        format_hypo_new(hypo_file, new_fn, lang2)
        compilation_rate = compile_code(new_fn, lang2, output_dir)
        print("Compilation rate: ", compilation_rate)
        return
    compilation_rates = []
    compiled_list = []
    for i in tqdm(range(len(eval_examples))):
        hypo_file = output_dir + out_fn + "." + str(i)
        new_fn = hypo_file + '.json'
        format_hypo_new(hypo_file, new_fn, lang2)
        compilation_rate = compile_code(new_fn, lang2, output_dir, False)
        compilation_rates.append(compilation_rate)
        if compilation_rate > 0:
            compiled_list.append(1)
        else:
            compiled_list.append(0)
    cr_mean = round(np.mean(compilation_rates), 2)
    cr_max = round(np.mean(compiled_list), 2)
    print("Compilation rate: ", cr_mean, cr_max)
    return 

def get_sample_generation_single(eval_dataloader, model, tokenizer, args, 
                      device, decoder_sid=None, temperature=0.5):
    pred_ids = []
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids,source_mask= batch                  
        with torch.no_grad():
            if args.model_type == 'roberta':
                    preds = model(source_ids=source_ids, source_mask=source_mask)
            else:
                if decoder_sid:
                    preds = model.generate(source_ids,
                                           attention_mask=source_mask,
                                           use_cache=True,
                                           num_beams=1,
                                           do_sample=True,
                                           temperature=temperature,
                                           early_stopping=False, # 如果是summarize就设为True
                                           max_length=args.max_target_length,
                                           decoder_start_token_id=tokenizer.sep_token_id)
                else:
                    preds = model.generate(source_ids,
                                           attention_mask=source_mask,
                                           use_cache=True,
                                           num_beams=1,
                                           do_sample=True,
                                           temperature=temperature,
                                           early_stopping=False, # 如果是summarize就设为True
                                           max_length=args.max_target_length)
    #                                        decoder_start_token_id=tokenizer.sep_token_id)
                top_preds = list(preds.cpu().numpy())
                pred_ids.extend(top_preds)
    hypos = [tokenizer.decode(pred, skip_special_tokens=True, 
                              clean_up_tokenization_spaces=False)
                              for pred in pred_ids]
    return hypos

def get_beam_generation_single(eval_dataloader, model, tokenizer, args, device):
    hypos=[]
    pred_ids = []
    for batch in tqdm(eval_dataloader,total=len(eval_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        source_ids,source_mask= batch                  
        with torch.no_grad():
            if args.model_type == 'roberta':
                    preds = model(source_ids=source_ids, source_mask=source_mask)
            else:
                preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=False, # 如果是summarize就设为True
                                       max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
                pred_ids.extend(top_preds)

        hypos = [tokenizer.decode(id, skip_special_tokens=True, 
                                  clean_up_tokenization_spaces=False)
                                  for id in pred_ids]
    return hypos



