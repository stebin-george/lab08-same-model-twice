import pytest

from bikes_data import make_rentals


@pytest.fixture
def rentals():
    return make_rentals()
