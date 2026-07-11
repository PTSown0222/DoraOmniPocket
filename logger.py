import logging
import sys
import os
from datetime import datetime
import colorlog

def setup_logger(save = False):
    """set up a log file"""

    log_dir = 'log'

    # check file logging storage
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # basic logs
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # time - model_name - levels - information
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'

        # set up color for log to human readable infor
        color_formatter = colorlog.ColoredFormatter(
            f"%(log_color)s{log_format}",
            datefmt=date_format,
            log_colors={
                'DEBUG': 'cyan', 'INFO': 'green',
                'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'bold_red',
            }
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(color_formatter)
        logger.addHandler(console_handler)

        # save logs
        if save:
            file_handler = logging.FileHandler(f"{log_dir}/training_{datetime.now().strftime('%Y%m%d')}.log")
            file_formatter = logging.Formatter(log_format, datefmt=date_format)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    
    return logger

