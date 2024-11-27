import sys
import pytest 
sys.path.append('../../closetx-app')
from app import create_app 


def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
