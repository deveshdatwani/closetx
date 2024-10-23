# this module will serve as a config for getting error codes so as to make the main code cleaner lookingfrom pydantic import BaseModel
from pydantic import BaseModel


class ResponseString(BaseModel):
    error: str =  "ERROR"
    redirecting: str = "REDIRECTING"
    login_success: str = "LOGIN SUCCESS"
    incorrect_password: str = "INCORRECT PASSWORD"
    no_username_or_password: str = "USERNAME AND PASSWORD NOT SUBMITTED"
    welcome_to_closetx: str = "GET WELCOME TO CLOSETX"
    user_deleted_succesfully: str = "USER DELETED SUCCESSFULLY"
    something_went_wrong: str = "SOMETHING WENT WRONG"
    registered_user_successfully: str = "200 SUCCESSFULLY REGISTERED USER"


class hell():
    def __init__(self):
        self.value = None