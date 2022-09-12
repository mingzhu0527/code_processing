import multiprocessing
import sys
import functools
import collections
from random import shuffle
import os
import traceback
import time


class InterpreterError(Exception): pass

def my_exec(cmd, globals=None, locals=None, description='source string'):
    try:
#         exec(cmd, globals, locals)
        exec(cmd)
    except SyntaxError as err:
        error_class = err.__class__.__name__
        detail = err.args[0]
        line_number = err.lineno
    except Exception as err:
        error_class = err.__class__.__name__
        detail = err.args[0]
        cl, exc, tb = sys.exc_info()
        line_number = traceback.extract_tb(tb)[-1][1]
    else:
        return
    error_str = "%s at line %d of %s: %s" % (error_class, line_number, description, detail)
    return error_str
#     raise InterpreterError("%s at line %d of %s: %s" % (error_class, line_number, description, detail))

class QueueStream:
    @classmethod
    def patch(cls, stream_name, queue):
        """Patch a standard output stream with a QueueStream"""
        setattr(sys, stream_name, cls(stream_name, queue))

    def __init__(self, stream_name, queue):
        self.stream_name = stream_name
        self.queue = queue

    def write(self, *args, **kwargs):
        self.queue.put((self.stream_name, args, kwargs))

    def flush(self, *args, **kwargs):
        """For file-like object API compatibility"""
        pass
    @staticmethod
    def drain(queue):
        """Drain a queue and print its messages to standard output streams"""
        while not queue.empty():
            stream, args, kwargs = queue.get()
            output_stream = getattr(sys, stream)
#             print(1, *args, **kwargs)
            output_stream.write(*args, **kwargs)
    

def worker_test(sec, output_queue=None):
    pid = os.getpid()
    ppid = os.getppid()
    # If we're a child process, replace standard output streams
    if pid != ppid and output_queue:
        QueueStream.patch("stdout", output_queue)
        QueueStream.patch("stderr", output_queue)
    
    # worker code
    print(f'Sleeping for {sec} second(s)')
    time.sleep(sec)
    print(f'Done sleeping')
    return sec

def worker_parallel_test(n):

    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()
    async_results = []
    secs = [x for x in range(n)]
    shuffle(secs)
    print(secs)
    for i, sec in enumerate(secs):
        output_queue = manager.Queue()
        call_fetch = functools.partial(
            worker_test, sec, output_queue=output_queue
        )
        async_results.append((output_queue, pool.apply_async(call_fetch)))
    pool.close()

    # Now print output in the order it was submitted
    for output_queue, async_result in async_results:
        while True:
            try:
                # this is the return value of worker function
                result = async_result.get(1)
                
            except multiprocessing.TimeoutError:
                # this raises while the process is still running.
                QueueStream.drain(output_queue)
                continue
            else:
                # if the process completed, drain and move to the next one.
                QueueStream.drain(output_queue)
                break

    # Wait for the work to complete.
    pool.join()
    return

def worker_command(pp, lang, i, output_queue=None):
    pid = os.getpid()
    ppid = os.getppid()
    # If we're a child process, replace standard output streams
    if pid != ppid and output_queue:
        QueueStream.patch("stdout", output_queue)
        QueueStream.patch("stderr", output_queue)
    result = exec_single_code_util(pp, lang, i, 3)
    print(result[0])
    return {"code_string":pp, "output":result}

def worker_parallel_command(n):

    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()
    async_results = []
    for i, pp in enumerate(programs_list[:2]):
        output_queue = manager.Queue()
        call_fetch = functools.partial(
            worker_command, pp, "Python", i, output_queue=output_queue
        )
        async_results.append((output_queue, pool.apply_async(call_fetch)))
    pool.close()

    # Now print output in the order it was submitted
    for output_queue, async_result in async_results:
        while True:
            try:
                # this is the return value of worker function
                result = async_result.get(1)
                print("*"*10)
            except multiprocessing.TimeoutError:
                # this raises while the process is still running.
                QueueStream.drain(output_queue)
                continue
            else:
                # if the process completed, drain and move to the next one.
                QueueStream.drain(output_queue)
                break

    # Wait for the work to complete.
    pool.join()
    return

def worker_exec(pp, output_queue=None):
    pid = os.getpid()
    ppid = os.getppid()
    # If we're a child process, replace standard output streams
    if pid != ppid and output_queue:
        QueueStream.patch("stdout", output_queue)
        QueueStream.patch("stderr", output_queue)
    result = run_exec(pp)
    return {"code_string":pp, "error":result}


def worker_parallel_exec(programs_list):

    pool = multiprocessing.Pool()
    manager = multiprocessing.Manager()
    async_results = []
    for i, pp in enumerate(programs_list[:15]):
        output_queue = manager.Queue()
        call_fetch = functools.partial(
            worker_exec, pp, output_queue=output_queue
        )
        async_results.append((output_queue, pool.apply_async(call_fetch)))
    pool.close()

    # Now print output in the order it was submitted
    for output_queue, async_result in async_results:
        while True:
            try:
                # this is the return value of worker function
                result = async_result.get(1)
                print(result)
                print("*"*10)
            except multiprocessing.TimeoutError:
                # this raises while the process is still running.
                QueueStream.drain(output_queue)
                continue
            else:
                # if the process completed, drain and move to the next one.
                QueueStream.drain(output_queue)
                break

    # Wait for the work to complete.
    pool.join()
    return

def abortable_worker(func, *args, **kwargs):
    timeout = kwargs.get('timeout', None)
    p = ThreadPool(1)
    res = p.apply_async(func, args=args)
    try:
        out = res.get(timeout)  # Wait timeout seconds for func to complete.
        return out
    except multiprocessing.TimeoutError:
        print("Aborting due to timeout")
        raise

def test(sec):
    worker_parallel_test(sec)

def run(programs_list):
    worker_parallel_exec(programs_list)
    


import time
import concurrent

# start = time.perf_counter()
# worker_parallel_exec(programs_list)
# end = time.perf_counter()
# print(f'Finished in {round(end-start, 2)} second(s)')