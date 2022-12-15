'''
@File         :utils.py
@Description  :
@Time         :2022/12/14 15:36:37
@Author       :tangs
@Version      :1.0
'''

import logging
from functools import wraps

LOG_FORMAT = '%(asctime)s.%(msecs)03d %(levelname)s %(process)d %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
logger.addHandler(handler)


def log_filter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(
            f"=============  Begin: {func.__module__+'/'+func.__name__}  ============="
        )
        func(*args, **kwargs)
        logger.info(
            f"=============   End: {func.__module__+'/'+func.__name__}   =============\n"
        )

    return wrapper
