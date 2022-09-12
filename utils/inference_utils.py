from run import *
from program_utils import *

def inference_prepro(lang1, lang2, model_type, device, src_codes, tgt_codes, function_data_path=None, 
                     tag='test', exp_suffix="_translation_exec_function/"):
    model, tokenizer, decoder_sid = get_model_by_name(model_type, device, lang1, lang2, exp_suffix)
    test_file, args = prep_eval_args(model_type, tokenizer, lang1, lang2, function_data_path, tag)
    args.eval_batch_size = 32
    if test_file:
        eval_examples, eval_features = get_eval_examples_from_file(test_file, tokenizer, args)
    else:
        eval_examples, eval_features = get_eval_examples(src_codes, tgt_codes, tokenizer, args)
    eval_dataloader = get_eval_dataloader(eval_features, args.eval_batch_size)
    return eval_examples, eval_features, eval_dataloader, model, tokenizer, args, decoder_sid

def read_example(code_string):
    examples=[]
    examples.append(
            Example(
                    idx = 0,
                    source=code_string.strip(),
                    target="",
                    ) 
            )
    return examples

def read_examples_from_list(src_codes, target_codes):
    examples=[]
    if len(target_codes) == 0:
        target_codes = [""]*len(src_codes)
    for i, (src, tgt) in enumerate(zip(src_codes, target_codes)):
        examples.append(Example(idx = i,
                        source=src.strip(),
                        target=tgt.strip()))
    return examples

def get_eval_tensors(code_string, tokenizer, args):
    eval_examples = read_example(code_string)
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
    return all_source_ids, all_source_mask
    
def get_testfile(lang1, lang2, data_path, tag='test'):
    test_src_file, test_tgt_file = get_eval_files(lang1, lang2, data_path, tag)
    test_file = test_src_file + "," + test_tgt_file
    return test_file

def get_eval_data_by_pid(eval_examples, eval_features, pids, reverse_map_dict, eval_batch_size):
    inds = [reverse_map_dict[x] for x in pids]
    selected_eval_examples = [eval_examples[x] for x in inds]
    selected_eval_features = [eval_features[x] for x in inds]
    eval_dataloader = eval_dataloader = get_eval_dataloader(selected_eval_features, eval_batch_size)
    return selected_eval_examples, eval_dataloader

def get_output_dir(model_type, lang1, lang2):
    output_dir = './sample_results/' + model_type + "/" + lang1 + "-" + lang2 + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def prep_eval_args(model_type, tokenizer, lang1, lang2, function_data_path=None, tag='test'):
    test_file = None
    if function_data_path:
        test_file = get_testfile(lang1, lang2, function_data_path, tag)
    output_dir = get_output_dir(model_type, lang1, lang2)
    args = set_func_args(model_type, test_file, output_dir)
    return test_file, args

def get_eval_examples_from_file(test_file, tokenizer, args, num_datapoints=-1):
    eval_examples = read_examples(test_file)
    if num_datapoints > 0:
        eval_examples = eval_examples[:num_datapoints]
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
    return eval_examples, eval_features

def get_eval_examples(src_codes, tgt_codes, tokenizer, args, num_datapoints=-1):
    eval_examples = read_examples_from_list(src_codes, tgt_codes)
    if num_datapoints > 0:
        eval_examples = eval_examples[:num_datapoints]
    eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
    return eval_examples, eval_features

def get_eval_dataloader(eval_features, eval_batch_size):
    all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)    
    eval_data = TensorDataset(all_source_ids,all_source_mask)   

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)
    return eval_dataloader

def get_eval_data(test_file, tokenizer, args, num_datapoints=-1):
    eval_examples, eval_features = get_eval_examples_from_file(test_file, num_datapoints)
    eval_dataloader = get_eval_dataloader(eval_features, args.eval_batch_size)
    return eval_examples, eval_dataloader


def get_model_by_name(model_type, device, lang1, lang2, exp_suffix="_translation_exec_function/"):
    
    model_name_or_path = model_name_dict[model_type]
    model_path = func_model_path + model_type + "/" + model_type + exp_suffix
    load_model_path_prefix = model_path + lang1 + "-" + lang2 + '/'
    load_model_path = load_model_path_prefix + "checkpoint-best-bleu/pytorch_model.bin"
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
    decoder_sid = tokenizer.eos_token_id
    if model_type == "codet5":
        decoder_sid = None
    model = model_class.from_pretrained(model_name_or_path)
    model.load_state_dict(torch.load(load_model_path))
    model.eval()
    model.to(device)
    return model, tokenizer, decoder_sid

def set_func_args(model_type, test_file, output_dir):
    parser = get_args_parser()
    model_name_or_path = model_name_dict[model_type]
    sys.argv = ['pl', '--do_test', 
    '--model_type', model_type, 
    '--model_name_or_path', model_name_or_path, 
    '--config_name', model_name_or_path, 
    '--tokenizer_name', model_name_or_path, 
    '--test_filename', test_file,
    '--output_dir', output_dir, 
    '--max_source_length', '200', 
    '--max_target_length', '300', 
    '--beam_size', '5', 
    '--eval_batch_size', '16']
    args = parser.parse_args()
    return args

def get_args_parser():
    parser = argparse.ArgumentParser()
    ## Required parameters  
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base" )
    parser.add_argument("--tokenizer_name", default="", required=True,
                        help="Pretrained tokenizer name or path if not the same as model_name")    
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str, 
                        help="Path to trained model: Should contain the .bin files" )    
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filenames (source and target files).")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. (source and target files).")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. (source and target files).")  

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available") 

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")   
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    return parser
# print arguments

func_model_path = home_path + "CodeXGLUE/Code-Code/code-to-code-trans/code/experiments/"

file_extensions = {"Java": ".java", "C++": ".cpp", "C": ".c", "Python": ".py","Javascript": ".js",
                   "PHP":".php", "C#":".cs"}
model_name_dict = {'codet5':"Salesforce/codet5-base", 
                   'plbart':"uclanlp/plbart-python-en_XX", 
                   'codebert':"microsoft/codebert-base"}
# model_path = "./plbart_official/"