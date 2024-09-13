# log_config.py
import logging
import sys

LOG_FORMAT = '%(asctime)s [%(name)s] %(levelname)s: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

def configure_logging(log_path, logger_name):
    # 创建一个日志记录器
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # 设置日志级别

    # 创建一个文件处理器
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)  # 设置文件处理器的日志级别
    file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    file_handler.setFormatter(file_formatter)

    # 创建一个控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)  # 设置控制台处理器的日志级别
    console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_handler.setFormatter(console_formatter)

    # 将处理器添加到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger