import subprocess
import sys
import time
from config import NUM_CLIENTS

print(f"Launching {NUM_CLIENTS} clients...")

processes = []

for cid in range(NUM_CLIENTS):
    p = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            f"from client import start_client; start_client({cid})"
        ]
    )
    processes.append(p)
    time.sleep(0.5)

for p in processes:
    p.wait()
