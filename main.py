from queue import Queue
from threading import Thread

import src.scraping as scraping


def main():
    base_url = scraping.configure()

    page_numbers = scraping.get_pages_number(base_url)

    saving_queue = Queue()

    saving_task = Thread(target=html_saving_loop, args=(saving_queue,))
    saving_task.start()

    for page_number in range(1, page_numbers):
        page_url = base_url + str(page_number)
        cases_links = scraping.get_links_from_page(page_url)

        for case_link in cases_links:
            case_content_link = case_link.replace("details", "content")
            url = "http://orzeczenia.wroclaw-srodmiescie.sr.gov.pl/" + case_content_link

            case_html = scraping.get_case(url)
            last_slash_index = case_content_link.rfind('/')
            case_identifier = case_content_link[last_slash_index:]
            task_data = (case_html, case_identifier)
            saving_queue.put(task_data)

    saving_queue.join()
    saving_queue.put('END')


def html_saving_loop(queue: Queue):
    while True:
        task_data = queue.get()
        if task_data is None:
            continue
        elif task_data == 'END':
            break
        case_html = task_data[0]
        case_identifier = task_data[1]
        scraping.save_case_details(case_html, case_identifier)
        queue.task_done()


main()
