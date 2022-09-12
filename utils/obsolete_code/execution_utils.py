import os
import json
import re
import tempfile as tfile
import time
from subprocess import STDOUT, check_output, Popen, PIPE
import subprocess
import threading
from tqdm import tqdm
import shlex
import pickle
# from queue_stream_utils import *

def exec_single_code(args):
    return exec_single_code_util(*args)

def exec_single_code_util(example, lang, i, timeout=3):
    err, op, return_code = exec_prog(example, timeout)
    num_programs_compiled = 0
    row = {
            "idx" : i,
            "code_string": example,
            "error_string": err,
            "output": op,
            "return_code": return_code
            }
    if return_code == 0:
        num_programs_compiled += 1
    return row, num_programs_compiled



class Command:
    def __init__(self, cmd):
        self.cmd = cmd;
        self.cmd = shlex.split(cmd)
        self.process = None;
        self.err = None
        self.op = None
    def run(self, timeout):
        def worker():
            #print("Thread started...");
            self.process = subprocess.Popen(self.cmd, stdout=PIPE, stderr=PIPE, shell=False);
            
            error = [i.decode('utf-8') for i in self.process.stderr.readlines()]
            self.err = '\n'.join(error)
            output = [i.decode('utf-8') for i in self.process.stdout.readlines()]
            self.op = '\n'.join(output)
            self.process.communicate()
            #print("Thread finished.");
 
        # Start child thread
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout)
 
        # When timeout, join() would return but process will be alive, thus need to terminate.
        if thread.isAlive():
            #print("Terminating process...")
            self.process.kill()
            return None, None, -1
        return self.err, self.op, self.process.returncode
    
def remove_comments(string, lang):
    if lang == 'python':
        pattern = "('''[\s\S]*''')|(''[\s\S]*''')"
        string = re.sub(pattern, '', string)
        return re.sub(r'(?m)^ *#.*\n?', '', string)
    else:
        pattern = '\/\*[\s\S]*\*\/'
        pattern2 = '[^:]//.*|/\\*((?!=*/)(?s:.))+\\*/'
        string = re.sub(pattern, '', string)
        string = re.sub(pattern2, '', string)                                              
        return string
    
def get_var_names(test_case):
    splits = test_case.split("\n")
    names = []
    for split in splits:
        idx = split.find(" =")
        names.append(split[:idx])
    return names


def get_class_func_names(code_str):
    function_pat = "def (.*)\("
    class_pat = "class (Solution[0-9]?)\(object\):"
    
    classes = []
    functions = []
    code_str = remove_comments(code_str, "py")
    code_lines = code_str.split("\n")
    
    
    for i, line in enumerate(code_lines):
        if "Solution" in line:
            try:
                class_match = re.findall(class_pat, line)
                classes.append(class_match[0])
            
                sample_lines = "\n".join(code_lines[i+1:i+6])
                function_match = re.findall(function_pat, sample_lines)
                functions.append(function_match[0])
            except:
                continue    
    return classes, functions

def py_formatting(test_case):
    test_case["inputs"] = test_case["inputs"].replace("null", "None")
    test_case["inputs"] = test_case["inputs"].replace("true", "True")
    test_case["inputs"] = test_case["inputs"].replace("false", "False")
    test_case["output"] = test_case["output"].replace("null", "None")
    test_case["output"] = test_case["output"].replace("true", "True")
    test_case["output"] = test_case["output"].replace("false", "False")
    return test_case

def exec_prog(codestring, timeout):
    
    with open("test.py", "w+") as tf:
        tf.write(codestring)
        file_path="./test.py"
    
    cmd = "python2 " + file_path
    
    command = Command(cmd);
    
    err, op, return_code = command.run(timeout=timeout);
    
    if err:
        cmd = "futurize -n -w " + file_path
        command = Command(cmd);
    
        err, op, return_code = command.run(timeout=timeout);
    
        cmd = "python3 " + file_path

        command = Command(cmd);

        err, op, return_code = command.run(timeout=timeout);
    if return_code == -1:
        return err, op, return_code 
    else:
        return err.strip(), op.strip(), return_code


def test_python_solution(codestring, test_case, solution, function, timeout):
    
    test_case = py_formatting(test_case)
    
    codestring = codestring + "\nsol = " + solution + "()\n"
    codestring = codestring + test_case["inputs"] + "\n"
    codestring = codestring + "expected = " + test_case["output"] + "\n"
    
    var_names = get_var_names(test_case["inputs"])
    var_names = ", ".join(var_names)
    
    codestring = codestring + "if expected == sol." + function + "(" + var_names + "):\n"
    codestring += "\tprint(True)\n"
    codestring += "else:\n"
    codestring += "\tprint(False)"
    err, op, return_code = exec_prog(codestring, timeout)
    return err, op, return_code


def generate_python_output(codestring, test_case, solution, function, timeout):
    #print(test_case)
    #test_case = py_formatting(test_case)
    
    codestring = codestring + "\nsol = " + solution + "()\n"
    codestring = codestring + test_case["inputs"] + "\n"
    #codestring = codestring + "expected = " + test_case["Output"] + "\n"
    
    var_names = get_var_names(test_case["inputs"])
    var_names = ", ".join(var_names)
    #print(var_names)
    codestring = codestring + "print(sol." + function + "(" + var_names + "))\n"
    # codestring = codestring + "if expected == sol." + function + "(" + var_names + "):\n"
    # codestring += "\tprint(True)\n"
    # codestring += "else:\n"
    # codestring += "\tprint(False)"
    #print(codestring)
    err, op, return_code = exec_prog(codestring, timeout)
    return err, op, return_code


