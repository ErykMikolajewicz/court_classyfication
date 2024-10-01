from queue import Queue
from threading import Thread
import json
from pathlib import Path
import argparse

import src.scraping as scraping
from src.exceptions import NoJustificationPart
from src.loggers import scraping_logger, logging_levels

parser = argparse.ArgumentParser()
parser.add_argument("init_page", default=1, type=int, nargs='?')
parser.add_argument("--logging_level", default='INFO', nargs='?',
                    choices=logging_levels)
args = parser.parse_args()

scraping_logger.setLevel(args.logging_level)

config_path = Path('config/scraping.json')
with open(config_path) as config_file:
    config = json.load(config_file)


def main():
    scraping_logger.info('Started.')

    page_numbers = scraping.get_pages_number()

    saving_queue = Queue()

    saving_task = Thread(target=html_saving_loop, args=(saving_queue,), daemon=True)
    saving_task.start()

    scraping_logger.debug('Started html saving loop - external info.')

    for page_number in range(args.init_page, page_numbers + 1):
        scraping_logger.info(f'Strona: {page_number}/{page_numbers}')
        case_part_links = scraping.get_links_from_page(page_number)

        for case_part_link in case_part_links:
            case_html = scraping.get_case(case_part_link)

            last_slash_index = case_part_link.rfind('/')
            case_identifier = case_part_link[last_slash_index + 1:]

            task_data = (case_html, case_identifier)
            saving_queue.put(task_data)

    saving_queue.join()
    saving_queue.put('END')


def html_saving_loop(queue: Queue):
    scraping_logger.debug('Started html saving loop - internal info.')

    while True:
        task_data = queue.get()
        if task_data is None:
            continue
        elif task_data == 'END':
            break
        case_html = task_data[0]
        case_identifier = task_data[1]

        scraping_logger.debug(case_identifier)

        try:
            scraping.save_case_details(case_html, case_identifier)
        except NoJustificationPart:
            scraping_logger.warning(case_identifier)
        finally:
            queue.task_done()


main()
