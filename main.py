"""Entry point to running programs. It is important to run scripts from here, due to paths issues."""
from dotenv import load_dotenv; load_dotenv()
from typing import Literal

from prefect_flows import get_all_raw_html, prepare_data
from data_exploration.classes_counting import plot_classes_chart
from data_exploration.boundary_looking import make_lda_plot
from scikit_runs.bag_unknown import train_bag_with_unknown, validate_bag_with_unknown
from scikit_runs.bag_tf_idf import train_tf_idf, validate_tf_idf
from scikit_runs.online import online_training, online_validate
from scikit_runs.tree_with_bag import train_random_tree, validate_random_tree


def get_and_prepare_data(court_type: str):
    get_all_raw_html(court_type)
    prepare_data(court_type)


def explore_data(court_type: str, label_type: Literal["detailed", "general"]):
    plot_classes_chart(court_type, label_type)
    make_lda_plot(12_000, label_type)

def scikit_bag_with_unknown(label_type: Literal["detailed", "general"]):
    train_bag_with_unknown(label_type)
    validate_bag_with_unknown(label_type)


def scikit_tf_idf(label_type: Literal["detailed", "general"]):
    train_tf_idf(label_type)
    validate_tf_idf(label_type)


def scikit_online(label_type: Literal["detailed", "general"]):
    online_training(label_type)
    online_validate(label_type)


def scikit_tree(label_type: Literal["detailed", "general"]):
    train_random_tree(label_type)
    validate_random_tree(label_type)


if __name__ == "__main__a":
    get_and_prepare_data('precinct')
    prepare_data('precinct')
    explore_data('precint', 'general')
    scikit_bag_with_unknown('general')
    scikit_tf_idf('general')
    scikit_online('general')
    scikit_tree('general')
