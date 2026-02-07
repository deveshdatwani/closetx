from model.app import app
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)

@app.task
def add_two_numbers(x, y):
    logger.debug(f"Received numbers: {x}, {y}")
    logger.info(f"Adding numbers: {x}, {y}")
    result = x + y
    logger.info(f"Adding {x} and {y} to get {result}")
    return result