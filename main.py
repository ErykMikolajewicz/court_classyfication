import src.scraping as scraping


def main():
    base_url = scraping.configure()

    page_numbers = scraping.get_pages_number(base_url)

    for page_number in range(1, page_numbers):
        page_url = base_url + str(page_number)
        cases_links = scraping.get_links_from_page(page_url)

        for case_link in cases_links:
            case_content_link = case_link.replace("details", "content")
            url = "http://orzeczenia.wroclaw-srodmiescie.sr.gov.pl/" + case_content_link

            case_html = scraping.get_case(url)
            last_slash_index = case_content_link.rfind('/')
            case_identifier = case_content_link[last_slash_index:]

            scraping.save_case_details(case_html, case_identifier)


main()
