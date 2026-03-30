import logging
import os


'''Set up of the debug/error logger'''


def logger_setup():
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    os.makedirs("logs", exist_ok=True)
    file_handler = logging.FileHandler('logs/app.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger