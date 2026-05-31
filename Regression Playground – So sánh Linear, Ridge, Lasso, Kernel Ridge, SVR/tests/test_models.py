"""Tests cho model factory, train va predict."""

import pytest
from sklearn.model_selection import train_test_split

from src.data import make_linear_dataset
from src.models import get_models, predict_model, train_model


@pytest.mark.parametrize("model_name", list(get_models()))
def test_models_can_train_and_predict(model_name: str) -> None:
    x, y = make_linear_dataset(n_samples=80)
    x_train, x_test, y_train, _ = train_test_split(
        x, y, test_size=0.25, random_state=42
    )
    model = get_models()[model_name]
    train_model(model, x_train, y_train)
    predictions = predict_model(model, x_test)
    assert len(predictions) == len(x_test)

