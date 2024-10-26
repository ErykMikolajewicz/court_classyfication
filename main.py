from prefect import flow

from prefect_tasks.get_data import get_raw_html, get_justification
from prefect_tasks.preprocessing import make_word_counts
from prefect_tasks.ml_model import train_model, validate_model


@flow
def ml_workflow():
    get_raw_html('training')
    get_justification('training')
    make_word_counts('training')
    train_model()
    validate_model()


if __name__ == "__main__":
    ml_workflow()
