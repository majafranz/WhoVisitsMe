import logging
from src.utils.config import LOG_ROOT
import os
from colorlog import ColoredFormatter

if not os.path.exists(LOG_ROOT):
    os.makedirs(LOG_ROOT, exist_ok=True)

FORMAT_FILE = '%(levelname)-8s | line %(lineno)-4d | %(asctime)-8s | %(message)s'
FORMAT_STOUT = '%(log_color)s%(asctime)-8s | line %(lineno)-4d | %(message)s%(reset)s'
DATE_FMT = '%d-%b-%y %H:%M:%S'

logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)

stream = logging.StreamHandler()
stream.setLevel(logging.DEBUG)
stream.setFormatter(ColoredFormatter(fmt=FORMAT_STOUT,
                                     datefmt=DATE_FMT))

file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'logfile.log'), mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(fmt=FORMAT_FILE,
                                datefmt=DATE_FMT))

logger.addHandler(stream)
logger.addHandler(file_handler)