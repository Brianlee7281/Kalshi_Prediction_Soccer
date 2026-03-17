"""Root conftest — loads .env so all tests pick up API keys."""

from dotenv import load_dotenv

load_dotenv()
