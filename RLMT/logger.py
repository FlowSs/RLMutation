import logging
import sys


class Log:
    def __init__(self, logger_prefix: str):
        self.logger = logging.getLogger(logger_prefix)
        # avoid creating another logger if it already exists
        if len(self.logger.handlers) == 0:
            self.logger = logging.getLogger(logger_prefix)
            self.logger.setLevel(level=logging.INFO)
            formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            ch.setLevel(level=logging.DEBUG)

            self.logger.addHandler(ch)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warn(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
