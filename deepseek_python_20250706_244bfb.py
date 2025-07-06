# Sample cosmic signature detection
def detect_cosmic(file):
    with open(file, 'rb') as f:
        header = f.read(6)
        if header == b'cosmic':
            return True
    return False