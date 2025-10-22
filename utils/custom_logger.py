import logging

class CustomLogger:
    LOG_LEVELS = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    def __init__(self, log_level, logger_name):
        self.logger = self.logger_setup(log_level, logger_name)

    def logger_setup(self, log_level, logger_name):
        level = self.LOG_LEVELS.get(log_level.upper(), logging.INFO)  # Default logging.INFO

        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler()
            ]
        )

        # Surpress logging for other libs
        logging.getLogger("PIL").setLevel(logging.WARNING)

        return logging.getLogger(logger_name)
