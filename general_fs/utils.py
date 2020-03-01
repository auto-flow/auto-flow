from uuid import uuid4

def get_id():
    return str(uuid4().hex)