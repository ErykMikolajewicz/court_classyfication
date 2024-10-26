import json
from pathlib import Path

from prefect import task, get_run_logger
from bs4 import BeautifulSoup

import src.scraping as scraping


config_path = Path('config/scraping.json')
with open(config_path) as config_file:
    config = json.load(config_file)


@task
def get_raw_html(destined_set, init_page=1):
    prefect_logger = get_run_logger()

    page_numbers = scraping.get_pages_number()

    prefect_logger.debug('Started html saving loop - external info.')

    for page_number in range(init_page, page_numbers + 1):
        prefect_logger.info(f'Strona: {page_number}/{page_numbers}')
        case_part_links = scraping.get_links_from_page(page_number)

        for case_part_link in case_part_links:
            case_html = scraping.get_case(case_part_link)

            last_slash_index = case_part_link.rfind('/')
            case_identifier = case_part_link[last_slash_index + 1:]
            case_path = Path('data') / destined_set / 'raw' / case_identifier
            with open(case_path, 'w') as case_file:
                case_file.write(case_html)


@task
def get_justification(dataset_type):
    prefect_logger = get_run_logger()

    raw_html_path = Path('data') / dataset_type / 'raw'
    for raw_case_path in raw_html_path.iterdir():
        with open(raw_case_path, 'r') as raw_html_file:
            case_html = raw_html_file.read()
            case_identifier = raw_case_path.name

        prefect_logger.debug(case_identifier)

        case_html = BeautifulSoup(case_html, 'html.parser')
        main_content = case_html.find('div', {'class': 'single_result'})
        content_parts = main_content.find_all('div')

        justification_part: BeautifulSoup | None = None
        for part in content_parts:
            content_header = part.find('h2').get_text()

            prefect_logger.debug(f'Case part header: {content_header}')

            content_header = content_header.lower()
            if content_header.find('uzasadnienie') != -1:
                justification_part = part
                break

        if justification_part is None:
            prefect_logger.debug(f'Case parts: {content_parts}')
            prefect_logger.warning(case_identifier)

        justification_elements = justification_part.find_all('p', recursive=False)
        justification_text = '\n'.join(element.text for element in justification_elements)

        raw_case_storing = Path('data') / dataset_type / 'justification' / case_identifier
        with open(raw_case_storing, 'w') as raw_case_file:
            raw_case_file.write(justification_text)

