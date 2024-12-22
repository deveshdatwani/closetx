import sys
import pytest 
import importlib


def test_home(client):
    response = client.get('/app/')
    assert response.status_code == 200
    assert b"Closetx" in response.data


# def test_user_registration(client):
#     print("Testing")
#     response = client.post('/register', data={"username":"deveshdatwani",
#                                            "emailid":"deveshd@bolt6.ai",
#                                            "password":"password"})
#     assert response.status_code == 200
#     assert "User registered successfully" in response.data


# def test_user_login(client):
#     print("Testing")
#     response = client.post('/login', data={"username":"deveshdatwani",
#                                            "password":"password"})
#     assert response.status_code == 200
#     assert "Login success" in response.data and "Use details" in response.data 