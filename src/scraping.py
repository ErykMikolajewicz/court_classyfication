import json
from time import sleep

import requests
from bs4 import BeautifulSoup

from src.exceptions import NoJustificationPart


def configure():
    url_base = "https://orzeczenia.wroclaw-srodmiescie.sr.gov.pl/search/advanced/"

    with open("./config/scraping.json") as file:
        config = json.load(file)

    department = config['department']
    start_date = config['start_date']
    end_date = config['end_date']
    order_by = config['order_by']
    ordering = config['ordering']

    return (f"{url_base}$N/$N/15502550/{department}/$N/$N/$N/{start_date}/{end_date}/"
            f"$N/$N/$N/$N/$N/$N/{order_by}/{ordering}/")


def get_pages_number(url: str):
    parsed_html = get_and_parse_html(url)

    # I expected t in t-data-grid-pager mean top, but they are 2 objects with that class in html.
    pages_html = parsed_html.find('div', {'class': 't-data-grid-pager'}).find_all('a')

    last_page_number = pages_html[-1].text
    return int(last_page_number)


def get_links_from_page(url: str):
    parsed_html = get_and_parse_html(url)

    nodes_with_links = parsed_html.find_all(lambda tag: tag.name == 'a' and tag.parent.name == 'h4')
    links = [node['href'] for node in nodes_with_links]

    return links


def get_case(url: str):
    case_html = get_and_parse_html(url)
    return case_html


def save_case_details(case_html, case_identifier: str):
    main_content = case_html.find('div', {'class': 'single_result'})
    content_parts = main_content.find_all('div')

    justification_part = None
    for part in content_parts:
        content_header = part.find('h2').get_text()
        if content_header == 'UZASADNIENIE':
            justification_part = part
            break

    if justification_part is None:
        raise NoJustificationPart()

    justification_elements = justification_part.find_all('p', recursive=False)

    justification_text = '\n'.join(element.text for element in justification_elements)

    path = "./data/raw/" + case_identifier
    with open(path, 'w') as file:
        file.write(justification_text)


def get_and_parse_html(url: str):
    headers = {
        'Host': 'orzeczenia.wroclaw-srodmiescie.sr.gov.pl',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;'
                  'q=0.9,image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://duckduckgo.com/',
    }
    response = requests.get(url, headers=headers)
    sleep(0.98)

    text = response.text
    parsed_html = BeautifulSoup(text, 'html.parser')
    return parsed_html
