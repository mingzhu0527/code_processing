import sys
# sys.path.append('/home/mingzhu/CodeModel/CodeGen_cwd/terminal_reward/')
# from terminal_reward.terminal_compiler import TerminalCompiler
from execution_utils import get_file_signature, run_command, tmp_path
import os
from tqdm import tqdm
import json

def compile_code_string(programs, lang, keys=[], is_tqdm=True):
    if keys == []:
        keys = [i for i in range(len(programs))]
    compile_programs = []
    compile_programs_count = 0
    if is_tqdm:
        for k, program in tqdm(zip(keys, programs), total=len(keys)):
            comp_output, is_compiled = compile_single_code(program, lang, k)
            compile_programs_count += is_compiled
            compile_programs.append(comp_output)
    else:
        for k, program in zip(keys, programs):
            comp_output, is_compiled = compile_single_code(program, lang, k)
            compile_programs_count += is_compiled
            compile_programs.append(comp_output)
    compilation_rate = compile_programs_count/len(programs)
    return compile_programs, compilation_rate

def get_compilation_by_split(program_id_lang_dic, lang, keys):
    lang_dic = program_id_lang_dic[lang]
    programs = []
    detoked_programs = []
    for k in keys:
        if k in lang_dic:
            v = lang_dic[k]
            program = v['program_formatted']
            program_tokenized = " ".join(v['tokens'])
            program_detok = detok_format(program_tokenized, file_detokenizers[lang])
            programs.append(program)
            detoked_programs.append(program_detok)
    compile_programs, compilation_rate = compile_code_string(programs, lang, keys)
    compile_programs_detoked, compilation_rate_detoked = compile_code_string(detoked_programs, lang, keys)
    return compile_programs, compilation_rate, compile_programs_detoked, compilation_rate_detoked

def save_compilation_report(statistics_file_path, error_fn, error_messages):
    if not os.path.exists(statistics_file_path):
        os.makedirs(statistics_file_path)

    with open(os.path.join(statistics_file_path, error_fn), "w+") as f:
        for line in error_messages:
            f.write(json.dumps(line))
            f.write("\n")
    return

        
def compile_single_code(example, lang, i):
    error, output, did_compile = lang2compiler[lang].compile_code_string(example)
    num_programs_compiled = 0
    row = {}
    if lang == "PHP":
        if "No errors" in output:
            num_programs_compiled +=1
        else:
            row = {
                    "idx" : i,
                    "code_string": example,
                    "error_string": output
                    }
#             error_messages.append(row)
    else:
        if did_compile:
            num_programs_compiled+=1
        else:
            row = {
                    "idx" : i,
                    "code_string": example,
                    "error_string": error
                    }
#             error_messages.append(row)
    return row, num_programs_compiled
    

def compile_code(hypo_file, lang, statistics_file_path, is_tqdm=True):
#     with open(hypo_file, "r") as infile:
#         examples = infile.readlines()
    with open(hypo_file, "r") as infile:
        examples_dict = json.load(infile)
    error_messages = []
    num_programs_compiled = 0
    if is_tqdm:
        for i, example in tqdm(examples_dict.items()):
            row, compiled = compile_single_code(example, lang, i)
            error_messages.append(row)
            num_programs_compiled += compiled
    
    else:
        for i, example in examples_dict.items():
            row, compiled = compile_single_code(example, lang, i)
            error_messages.append(row)
            num_programs_compiled += compiled

    compilation_rate = num_programs_compiled/len(examples_dict)
#     print("Compilation Rate: ", compilation_rate)
#     statistics_file_path = os.path.join(stats_path, folder)    

    if not os.path.exists(statistics_file_path):
        os.makedirs(statistics_file_path)

    with open(os.path.join(statistics_file_path, hypo_file + '.error_report.jsonl'), "w+") as f:
        for line in error_messages:
            f.write(json.dumps(line))
            f.write("\n")
    return compilation_rate


def write_tmp_file(codestring, lang):
    moment = get_file_signature()
    fn_name = tmp_path + moment + file_extensions[lang]
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()
    return fn_name

def run_compile(codestring, lang, timeout=5):
    if lang == "Javascript":
        codestring = codestring.replace("document.write", "console.log")
    fn_name = write_tmp_file(codestring, lang)
    cmd = compile_commands[lang] + " " + fn_name 
    if lang in ['C', 'C++']:
        exec_file = fn_name + ".out"
        cmd = compile_commands[lang] + " " + fn_name + " -o " + exec_file
        output = run_command(cmd, timeout)
        if os.path.exists(exec_file):
            os.remove(exec_file)
    elif lang == "C#":
        exec_file = fn_name[:-3] + ".exe"
        cmd = compile_commands[lang] + " " + fn_name
        output = run_command(cmd, timeout)
        if os.path.exists(exec_file):
            os.remove(exec_file)
    else:    
        output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output

def run_compile_java(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.java'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()
    cmd = "javac " + fn_name
    output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output

def run_compile_cpp(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.cpp'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()
    exec_file = fn_name + ".out"
    cmd1 = "g++ " + fn_name + " -o " + exec_file
    compile_output = run_command(cmd1, timeout)
    os.remove(fn_name)
    if os.path.exists(exec_file):
        os.remove(exec_file)
    return compile_output

def run_compile_c(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.c'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()
    exec_file = fn_name + ".out"
    cmd1 = "gcc " + fn_name + " -o " + exec_file
    compile_output = run_command(cmd1, timeout)
    os.remove(fn_name)
    if os.path.exists(exec_file):
        os.remove(exec_file)
    return compile_output

def run_compile_csharp(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.cs'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()
    exec_file = tmp_path + moment + ".exe"
    cmd1 = "mcs " + fn_name
    compile_output = run_command(cmd1, timeout)
    os.remove(fn_name)
    if os.path.exists(exec_file):
        os.remove(exec_file)
    return compile_output

def run_compile_python3(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.py'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()

    cmd = "python3 -m py_compile " + fn_name
    output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output

def run_compile_python2(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.py'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()

    cmd = "python2 -m py_compile " + fn_name
    output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output

def run_compile_python(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.py'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()

    cmd = "python2 -m py_compile " + fn_name
    output = run_command(cmd, timeout)
    if 'error' in output:
        cmd = "python3 -m py_compile " + fn_name
        output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output

def run_compile_js(codestring, timeout=5):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.js'
    f = open(fn_name, 'w')
    new_codestring = codestring.replace("document.write", "console.log")
    f.write(new_codestring)
    f.close()

    cmd = "node " + fn_name
    output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output

def run_compile_php(codestring, timeout=2):
    moment = get_file_signature()
    fn_name = tmp_path + moment + '.php'
    f = open(fn_name, 'w')
    f.write(codestring)
    f.close()

    cmd = "php " + fn_name
    output = run_command(cmd, timeout)
    os.remove(fn_name)
    return output



file_extensions = {"Java": ".java", "C++": ".cpp", "C": ".c", "Python": ".py","Javascript": ".js",
                   "PHP":".php", "C#":".cs"}
compile_commands = {"Java": "javac", "C++": "g++", "C": "gcc", "Python": "python3 -m py_compile", "Python3": "python3", 
                    "Python2": "python2", "Javascript": "node", "PHP":"php", "C#":"csc"}
exec_commands = {"Java": "java", "C++": "g++", "C": "gcc", "Python": "python3", "Python3": "python3", 
                    "Python2": "python2", "Javascript": "node", "PHP":"php", "C#":"mcs"}

file_compilers = {"Java": run_compile_java, "C++": run_compile_cpp, "C": run_compile_c, "Python": run_compile_python,
                   "Javascript": run_compile_js, "PHP": run_compile_php, "C#": run_compile_csharp}
# compilation
#Use Python, C++, C#, C, Java, PHP as arguments
# lang2compiler = {
#                     "Python": TerminalCompiler('Python'),
#                     "C++": TerminalCompiler('C++'),
#                     "C": TerminalCompiler('C'),
#                     "C#": TerminalCompiler('C#'),
#                     "PHP": TerminalCompiler('PHP'),
#                     "Java": TerminalCompiler('Java')
#                 }