import uuid
import threading

def get_id():
    return str(uuid.uuid4()).replace('-', '')

lock = threading.Lock()

__all__ = [
    'get_id',
    'lock'
]