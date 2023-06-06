import logging
import os
import time
def create_logger(log_dir, model_name, phase="train"):
    logging.basicConfig(filename=log_dir, format=' %(message)s')
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(model_name, time_str, phase)
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file+'.txt'))
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console_handler)
    return logger