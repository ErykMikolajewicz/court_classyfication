import logging

logging_levels = logging.getLevelNamesMapping().keys()

logging.basicConfig()
scraping_logger = logging.getLogger('scraping_logger')
