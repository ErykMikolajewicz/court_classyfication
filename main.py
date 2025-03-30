"""Entry point to running programs. It is important to run scripts from here, due to paths issues."""
from typing import Literal

from dotenv import load_dotenv; load_dotenv()

from prefect_flows import get_all_raw_html, prepare_data
from data_exploration.classes_counting import plot_classes_chart
from data_exploration.boundary_looking import make_lda_plot
from scikit_runs.bag_unknown import bag_with_unknown
from scikit_runs.bag_tf_idf import train_tf_idf
from pytorch_runs.torch_basic_run import torch_train_bag_with_unknown2


def get_and_prepare_data(court_type: str):
    get_all_raw_html(court_type)
    prepare_data(court_type)


def explore_data(court_type: str, label_type: Literal["detailed", "general"]):
    plot_classes_chart(court_type, label_type)
    make_lda_plot(12_000, label_type)


def scikit_bag_with_unknown(label_type: Literal["detailed", "general"]):
    bag_with_unknown(label_type)


def scikit_tf_idf(label_type: Literal["detailed", "general"]):
    train_tf_idf(label_type)


def pytorch_run(label_type: Literal["detailed", "general"]):
    torch_train_bag_with_unknown2(label_type, batch_size=200)

if __name__ == "__main__":
    get_and_prepare_data('precinct')
    explore_data('precinct', 'general')
    scikit_bag_with_unknown('general')
    scikit_tf_idf('general')
    pytorch_run('general')
