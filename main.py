from prefect import flow

from prefect_tasks.get_data import get_data
from prefect_tasks.preprocessing import make_word_counts
from prefect_tasks.ml_model import train_model, validate_model


@flow
def ml_workflow():
    get_data()
    make_word_counts()
    train_model()
    validate_model()


if __name__ == "__main__":
    ml_workflow()
