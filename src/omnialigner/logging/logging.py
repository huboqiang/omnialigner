import logging

def setup_logger(logger_file="./omni_aligner.log", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger('omni_aligner')
    logger.setLevel(level)
    
    # Create file handler with timestamp format
    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(logger_file)
    file_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger
