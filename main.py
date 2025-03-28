"""Entry point to running programs. It is important to run scripts from here, due to paths issues."""
from dotenv import load_dotenv; load_dotenv()
from typing import Literal

from prefect_flows import get_all_raw_html, prepare_data, prepare_tokens
from data_exploration.classes_counting import plot_classes_chart
from data_exploration.boundary_looking import make_lda_plot
from scikit_runs.bag_unknown import train_bag_with_unknown, validate_bag_with_unknown
from scikit_runs.bag_tf_idf import train_tf_idf, validate_tf_idf
from scikit_runs.online import online_training, online_validate
from scikit_runs.tree_with_bag import train_random_tree, validate_random_tree
from scikit_runs.tokens_run import train_tokens, validate_tokens
from pytorch_runs.torch_basic_run import torch_train_bag_with_unknown


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


def token_flow(court_type: str):
    prepare_tokens(court_type)


def scikit_token(label_type: Literal["detailed", "general"]):
    train_tokens(label_type)
    validate_tokens(label_type)


def pytorch_run(label_type: Literal["detailed", "general"]):
    torch_train_bag_with_unknown(label_type)

if __name__ == "__main__":
    # get_and_prepare_data('precinct')
    # prepare_data('precinct')
    # explore_data('precinct', 'general')
    # scikit_bag_with_unknown('general')
    # scikit_tf_idf('general')
    # scikit_online('general')
    # scikit_tree('general')
    # token_flow('precinct')
    # scikit_token('general')
    pytorch_run('general')
