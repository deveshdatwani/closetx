import pytest
from user_app.app import create_app


@pytest.fixture()
def app():
    app = create_app('user_app.config.config.Config')
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()