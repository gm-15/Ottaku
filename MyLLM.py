import base64
import time
import urllib

import streamlit as st

from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")