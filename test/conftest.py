import os
import sys
import pytest 
sys.path.append(os.path.expanduser("~")+'/closetx')
import importlib  
foobar = importlib.import_module("closetx-app.app")


@pytest.fixture
def client():
    app = app.create_app()
    with app.test_client() as client:
        yield client