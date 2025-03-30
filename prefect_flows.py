import json

from prefect import flow
from prefect.task_runners import ThreadPoolTaskRunner
from prefect_ray import RayTaskRunner

from prefect_tasks.get_data import get_raw_html_from_court, get_justification
from prefect_tasks.preprocessing import make_words_counts, preprocessed_justification


@flow(task_runner=ThreadPoolTaskRunner(max_workers=10))
def get_all_raw_html(court_type):
    with open('./config/courts.json') as config_file:
        courts_list = config_file.read()

    all_courts = json.loads(courts_list)

    courts_one_type = all_courts[court_type]
    for court in courts_one_type:
        court['type'] = court_type

    get_raw_html_from_court.map(courts_one_type).wait()


@flow(task_runner=RayTaskRunner())
def prepare_data(court_type: str):
    with open('./config/courts.json') as config_file:
        courts_list = config_file.read()
    courts_list = json.loads(courts_list)

    courts_one_type = courts_list[court_type]
    preprocessings = []
    words_count = []
    preparations = []
    preprocessing_args = {}
    word_counts_args = {}
    for court in courts_one_type:
        appeal = court['appeal']
        court_name = court['name']

        preparation = get_justification.submit(court_type, appeal, court_name)
        task_id = preparation.task_run_id
        preprocessing_args[task_id] = (court_type, appeal, court_name)
        preparations.append(preparation)

    for preparation in preparations:
        preparation_id = preparation.task_run_id
        preparation.wait()
        next_args = preprocessing_args[preparation_id]
        preprocessing = preprocessed_justification.submit(*next_args)
        preprocessing_id = preprocessing.task_run_id
        word_counts_args[preprocessing_id] = next_args
        preprocessings.append(preprocessing)

    for preprocessing in preprocessings:
        preprocessing_id = preprocessing.task_run_id
        preprocessing.wait()
        next_args = word_counts_args[preprocessing_id]
        preprocessing = make_words_counts.submit(*next_args)
        words_count.append(preprocessing)

    for word_count in words_count:
        word_count.wait()
