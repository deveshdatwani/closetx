import functools
from flask import Blueprint, g, flash, redirect, render_template, request, session, url_for, current_app
from .lib.db_helper import * 
import numpy as np


auth = Blueprint("ml", __name__)
model = "PLACEHOLDER"


def get_match_score(top: bytes, bottom: bytes):
    embedding_top = model(top)
    embedding_bottom = model(bottom)
    score = np.dot(embedding_bottom, embedding_top)
    return score


def match_apparel(top: bytes, bottom: bytes):
    score = get_match_score(top, bottom)
    return score