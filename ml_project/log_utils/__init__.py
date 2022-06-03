import logging
import logging.config
import yaml


def init_loggers():
    with open('configs/log.conf.yaml', mode='r') as config_file:
        logging.config.dictConfig(yaml.safe_load(config_file))
