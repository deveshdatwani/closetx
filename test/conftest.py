import sys
import pytest 
sys.path.append('../closetx-app')
print(sys.path)
from app import create_app


@pytest.fixture
def client():
    app = create_app()
    with app.test_client() as client:
        yield client