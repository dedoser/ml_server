from dotenv import load_dotenv
from fastapi import FastAPI

from src.Controllers import conrtollers

app = FastAPI()

app.include_router(conrtollers)