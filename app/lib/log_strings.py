from pydantic import BaseModel


class LoggerStringTemplate(BaseModel):
    error = "ERROR"
    redirecting = "REDIRECTING"


class hell():
    def __init__(self):
        self.value = None