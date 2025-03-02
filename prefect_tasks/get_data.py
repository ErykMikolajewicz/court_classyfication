"""Functions to get raw case html from court site, and extract justification from them."""
from pathlib import Path
import re

from prefect import task, get_run_logger
from bs4 import BeautifulSoup
from prefect.tasks import exponential_backoff
from prefect.cache_policies import NO_CACHE

from src.scraping import CourtScraper


@task(retries=3, retry_delay_seconds=exponential_backoff(10), cache_policy=NO_CACHE)
def get_raw_html_from_page(court_scraper: CourtScraper, court_type: str, appeal_name: str, court_name: str,
                           page_number: int):
    case_links = court_scraper.get_links_from_page(page_number)

    for case_link in case_links:
        last_slash_index = case_link.rfind('/')
        case_identifier = case_link[last_slash_index + 1:]

        case_html = court_scraper.get_case_html(case_identifier)
        case_dir = Path('data') / 'raw' / court_type / appeal_name / court_name
        case_path = case_dir / case_identifier
        try:
            with open(case_path, 'w', encoding='utf-8') as case_file:
                case_file.write(case_html)
        except FileNotFoundError:
            case_dir.mkdir(parents=True)
            with open(case_path, 'w', encoding='utf-8') as case_file:
                case_file.write(case_html)

@task
def get_raw_html_from_court(court_config):
    prefect_logger = get_run_logger()

    court_type = court_config['type']
    url = court_config['url']
    appeal = court_config['appeal']
    court_name = court_config['name']

    court_scraper = CourtScraper(url)
    pages_number = court_scraper.get_pages_number()
    for page_number in range(1, pages_number + 1):
        prefect_logger.info(f'Page: {page_number}/{pages_number}')
        get_raw_html_from_page(court_scraper, court_type, appeal, court_name, page_number)


@task
def get_justification(court_type, appeal_name, court_name):
    prefect_logger = get_run_logger()

    raw_html_path = Path('data') / 'raw' / court_type / appeal_name / court_name
    case_justification_dir = Path('data') / 'justification' / court_type / appeal_name / court_name

    if not raw_html_path.exists(): # Occur when is no cases in court page, sometimes happen in regional courts
        return None

    justification_warning_counter = 0
    for raw_case_path in raw_html_path.iterdir():

        with open(raw_case_path, 'r', encoding='utf-8') as raw_html_file:
            case_html = raw_html_file.read()
        case_identifier = raw_case_path.name

        prefect_logger.debug(case_identifier)

        case_html = BeautifulSoup(case_html, 'lxml')

        justification_text = ''
        for tag in case_html.find_all('h2'):
            if tag.text == 'UZASADNIENIE':
                justification_text = '\n'.join(t.text for t in tag.find_next_siblings('p'))
                if not justification_text:
                    justification_text = '\n'.join(t.text for t in tag.find_all_next('p'))
                break

        if not justification_text:
            search_flag = re.search('.*UZASADNIENIE.*', justification_text, re.IGNORECASE)
            if search_flag:
                justification_warning_counter += 1
                prefect_logger.warning(f'Lack of justification for case: {case_identifier}'
                                       f' warning number: {justification_warning_counter}')
            continue

        case_justification_path = case_justification_dir / case_identifier
        try:
            with open(case_justification_path, 'w', encoding='utf-8') as raw_case_file:
                raw_case_file.write(justification_text)
        except FileNotFoundError:
            case_justification_dir.mkdir(parents=True)
            with open(case_justification_path, 'w', encoding='utf-8') as raw_case_file:
                raw_case_file.write(justification_text)
