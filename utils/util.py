#   Written by anhlbt
#   Last Update: 11/05/2021
#


from os.path import dirname, join, realpath, isfile
import sys

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


