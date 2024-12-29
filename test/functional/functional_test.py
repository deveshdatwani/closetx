import sys
import pytest 
import importlib


def test_home(client):
    response = client.get('/app/')
    assert response.status_code == 200
    assert b"Closetx" in response.data


def test_user_registration(client):
    response = client.post('/app/register', data={"username":"deveshdatwani",
                                           "email":"deveshd@bolt6.ai",
                                           "password":"password"})
    assert response.status_code == 200
    assert b"User registered successfully" in response.data


def test_user_login(client):
    print("Testing")
    response = client.post('/app/login', data={"username":"deveshdatwani",
                                           "password":"password"})
    assert response.status_code == 200
    assert b"Login success"   in response.data and b"details" in response.data 


def test_user_deletion(client):
    response = client.delete('/app/delete', data={"username":"deveshdatwani",
                                           "password":"password"})
    assert response.status_code == 200
    assert b"User deleted" in response.data