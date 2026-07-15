import os
from dotenv import load_dotenv

load_dotenv()


def _bool_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ('1', 'true', 'yes', 'on')


def _int_env(name, default):
    value = os.environ.get(name)
    if value is None or value.strip() == '':
        return default
    return int(value)


DEBUG = _bool_env('DEBUG', True)
HOST = os.environ.get('HOST', '127.0.0.1')
PORT = _int_env('PORT', 5000)

DEFAULT_EXCHANGE = os.environ.get('DEFAULT_EXCHANGE', 'NASDAQ')
COLLECT_DATA_TIMEOUT_SECONDS = _int_env('COLLECT_DATA_TIMEOUT_SECONDS', 15)
CACHE_TTL_SECONDS = _int_env('CACHE_TTL_SECONDS', 15 * 60)
MIN_DATA_POINTS = _int_env('MIN_DATA_POINTS', 30)
MAX_DATA_POINTS = _int_env('MAX_DATA_POINTS', 500)
