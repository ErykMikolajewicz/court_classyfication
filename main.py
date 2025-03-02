"""Entry point to running programs. It is important to run scripts from here, due to paths issues."""
from dotenv import load_dotenv; load_dotenv()

from prefect_flows import get_all_raw_html, prepare_data
from data_exploration.classes_counting import plot_classes_chart
from data_exploration.boundary_looking import make_lda_plot
from scikit_runs.bag_unknown import train_bag_with_unknown, validate_bag_with_unknown
from scikit_runs.bag_tf_idf import train_tf_idf, validate_tf_idf
from scikit_runs.online import online_training, online_validate


def explore_data():
    plot_classes_chart()
    make_lda_plot(12_000)

def scikit_bag_with_unknown():
    train_bag_with_unknown()
    validate_bag_with_unknown()


def scikit_tf_idf():
    train_tf_idf()
    validate_tf_idf()


def scikit_online():
    online_training()
    online_validate()

if __name__ == "__main__":
    get_all_raw_html('precinct')
    prepare_data('precinct')
    explore_data()
    scikit_bag_with_unknown()
    scikit_tf_idf()
    scikit_online()
