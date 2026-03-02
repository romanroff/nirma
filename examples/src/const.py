from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('DEEPSEEK_API_KEY')
LIGHTRAG_API = os.getenv('LIGHTRAG_API')

__all__=[
    'API_KEY',
    'LIGHTRAG_API'
]