import sys
import pytest 
import importlib


def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"hacked" in response.data
