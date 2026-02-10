import logging
import functools
from fastapi import HTTPException
logger = logging.getLogger(__name__)

def api_error_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("api_handler_exception")
            raise HTTPException(status_code=500, detail=f"internal server error: {e}")
    return wrapper

def task_error_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            logger.exception("task_exception")
            raise
    return wrapper
