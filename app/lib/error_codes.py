# this module will serve as a config for getting error codes so as to make the main code cleaner lookingfrom pydantic import BaseModel
from pydantic import BaseModel


class LoggerStringTemplate(BaseModel):
    error: str =  "ERROR"
    redirecting: str = "REDIRECTING"


class hell():
    def __init__(self):
        self.value = None