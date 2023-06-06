import logging

def create_logger(log_dir):
    logging.basicConfig(filename=log_dir, format=' %(message)s')
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console_handler)
    return logger