import json
from time import sleep
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from prefect import get_run_logger


config_path = Path('config/scraping.json')
with open(config_path) as config_file:
    config = json.load(config_file)

url_authority = config['url_authority']
court_number = config['court_number']
department = config['department']
start_date = config['start_date']
end_date = config['end_date']
order_by = config['order_by']
ordering = config['ordering']
destined_set = config['destined_set']

search_url = (f'https://{url_authority}/search/advanced/$N/$N/{court_number}/{department}/$N/$N/$N/'
              f'{start_date}/{end_date}/$N/$N/$N/$N/$N/$N/{order_by}/{ordering}/')


def get_pages_number() -> int:
    prefect_logger = get_run_logger()
    prefect_logger.debug(f'Url to search: {search_url}')

    html_text = get_html(search_url)
    parsed_html = BeautifulSoup(html_text, 'html.parser')

    # I expected t in t-data-grid-pager mean top, but they are 2 objects with that class in html.
    pages_html = parsed_html.find('div', {'class': 't-data-grid-pager'}).find_all('a')
    prefect_logger.debug(f'Html with pages number: {pages_html}')

    last_page_number = pages_html[-1].text
    return int(last_page_number)


def get_links_from_page(page_number: int) -> list[str]:
    html_text = get_html(search_url + str(page_number))
    parsed_html = BeautifulSoup(html_text, 'html.parser')

    nodes_with_links = parsed_html.find_all(lambda tag: tag.name == 'a' and tag.parent.name == 'h4')
    links = [node['href'] for node in nodes_with_links]

    return links


def get_case(case_part_link: str):
    case_content_link = 'https://' + url_authority + case_part_link.replace("details", "content")
    case_html = get_html(case_content_link)
    return case_html


def get_html(url: str) -> str:
    prefect_logger = get_run_logger()

    headers = {
        'Host': url_authority,
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;'
                  'q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://duckduckgo.com/',
    }
    response = requests.get(url, headers=headers)

    prefect_logger.debug(f'{url} \n Status code: {response.status_code}')

    sleep(0.98)

    html_text = response.text
    return html_text
