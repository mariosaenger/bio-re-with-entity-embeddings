import os
import logging

from logging import Logger
from datetime import datetime

LOG_ROOT_DIR = "_logs"
if not os.path.exists(LOG_ROOT_DIR):
    os.makedirs(LOG_ROOT_DIR)

LOG_FILE = os.path.join(LOG_ROOT_DIR, datetime.utcnow().strftime("%Y%m%d%H%M%S.log"))


class LogUtil(object):

    @staticmethod
    def create_logger(name: str, level=logging.DEBUG) -> Logger:
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        logger = logging.getLogger(name)
        logger.setLevel(level)

        if logger.hasHandlers():
            return logger

        std_handler = logging.StreamHandler()
        std_handler.setLevel(level)
        std_handler.setFormatter(log_format)
        logger.addHandler(std_handler)

        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

        return logger


class LoggingMixin(object):

    def __init__(self, logger_name: str = None):
        logger_name = logger_name if logger_name else self.__class__.__name__
        self.logger = LogUtil.create_logger(logger_name)

    def log_debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def log_info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def log_warn(self, msg, *args, **kwargs):
        self.logger.warn(msg, *args, **kwargs)

    def log_error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)
