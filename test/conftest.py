import sys
import pytest 
sys.path.append('../closetx-app')
from app import create_app


@pytest.fixture
def client():
    app = create_app()
    with app.test_client() as client:
        yield client