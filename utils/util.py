#   Written by anhlbt
#   Last Update: 11/05/2021
#

from itertools import chain
from functools import reduce
from socket import gethostname
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from contextlib import redirect_stdout
from timeit import default_timer as timer
from multiprocessing.pool import ThreadPool
from os.path import dirname, join, realpath, isfile
import sys
from typing import Callable



C_DIR = dirname(realpath(__file__))
P_DIR = dirname(C_DIR)

sys.path.insert(0, P_DIR)
import numpy as np

import logging
# import redis
import time


def exception_logger(get_logger="exception_logger", file_logger= "exception.log"):
    """
    Creates a logging object and returns it
    """
    logger = logging.getLogger(get_logger)
    logger.setLevel(logging.INFO)
 
    # create the logging file handler
    fh = logging.FileHandler(file_logger, mode='a')
 
    # fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    # formatter = logging.Formatter(fmt)
    formatter = logging.Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%Y-%m-%d %H:%M:%S')

    fh.setFormatter(formatter)
 
    # add handler to logger object
    logger.addHandler(fh)
    return logger   

logger = exception_logger(get_logger="logger", file_logger= join(P_DIR, "logs/go2joy.log"))

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            logger.info("Elapsed time of '%s': %s seconds" % (method.__name__, te - ts))
        return result
    return timed


def parallel(func=None, args=(), merge_func=lambda x:x, parallelism = cpu_count()):
    def decorator(func: Callable):
        def inner(*args, **kwargs):
            results = Parallel(n_jobs=parallelism)(delayed(func)(*args, **kwargs) for i in range(parallelism))
            return merge_func(results)
        return inner
    if func is None:
        # decorator was used like @parallel(...)
        return decorator
    else:
        # decorator was used like @parallel, without parens
        return decorator(func)