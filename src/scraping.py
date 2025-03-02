import json
from time import sleep
from pathlib import Path
import re

import httpx
from bs4 import BeautifulSoup
from prefect import get_run_logger

from src.exceptions import InvalidStatusCode

config_path = Path('config/scraping.json')
with open(config_path) as config_file:
    config = json.load(config_file)

department = config['department']
start_date = config['start_date']
end_date = config['end_date']
order_by = config['order_by']
ordering = config['ordering']


class CourtScraper:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.search_url = (f'{base_url}search/advanced/$N/$N/$N/{department}/$N/$N/$N/'
                           f'{start_date}/{end_date}/$N/$N/$N/$N/$N/$N/{order_by}/{ordering}/')

        http_index = self.base_url.find('//')
        url_authority = self.base_url[http_index + 2 : -1]

        headers = {
            'Host': url_authority,
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0',
            'Accept': 'text/html,application/xhtml+xml,application/xml;'
                      'q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://duckduckgo.com/',
        }

        self.client = httpx.Client(http2=True, follow_redirects=True, headers=headers)

    def get_pages_number(self) -> int:
        prefect_logger = get_run_logger()
        prefect_logger.debug(f'Url to search: {self.search_url}')

        invalid_status_counter = 0
        while True:
            try:
                html_text = self.__get_html(self.search_url)
                break
            except InvalidStatusCode as e:
                invalid_status_counter += 1
                if invalid_status_counter == 3:
                    raise e

        parsed_html = BeautifulSoup(html_text, 'lxml')

        # I expected t in t-data-grid-pager mean top, but they are 2 objects with that class in html.
        pages_in_pagination = parsed_html.find('div', {'class': 't-data-grid-pager'})
        if pages_in_pagination is None:
            sleep(10) # to do not make to many requests
            return 0
        pages_links = pages_in_pagination.find_all('a')

        prefect_logger.debug(f'Html with pages number: {pages_links}')

        last_page_number = pages_links[-1].text
        return int(last_page_number)

    def get_links_from_page(self, page_number: int) -> list[str]:
        html_text = self.__get_html(self.search_url + str(page_number))
        parsed_html = BeautifulSoup(html_text, 'html.parser')

        nodes_with_links = parsed_html.find_all(lambda tag: tag.name == 'a' and tag.parent.name == 'h4')
        links = [node['href'] for node in nodes_with_links]

        return links

    def get_case_html(self, case_identifier: str):
        case_content_link = f'{self.base_url}/content/$N/{case_identifier}'
        case_html = self.__get_html(case_content_link)
        return case_html

    def __get_html(self, url: str) -> str:
        prefect_logger = get_run_logger()

        response = self.client.get(url)

        if response.status_code != 200:
            prefect_logger.warning(f'{url} \n Status code: {response.status_code}')
            raise InvalidStatusCode(f'{url} \n Status code: {response.status_code}')

        sleep(1)

        html_text = response.text

        self.__search_captcha(html_text)

        return html_text

    @staticmethod
    def __search_captcha(html):
        captcha_text = ('.*Wykryliśmy zbyt dużą liczbę zapytań pochodzących '
                        'z tego adresu, proszę wprowadzić kod z obrazka.*')

        captcha_regex = re.compile(captcha_text)

        case_html = BeautifulSoup(html, 'lxml')

        nodes = case_html.find_all('h3')

        for node in nodes:
            if captcha_regex.search(node.text):
                raise Exception('Captcha block access!')
