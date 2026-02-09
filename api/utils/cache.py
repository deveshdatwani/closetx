import os

BASE = os.getenv("CACHE_DIR","./cache")
RAW = f"{BASE}/raw"
PROC = f"{BASE}/processed"

os.makedirs(RAW, exist_ok=True)
os.makedirs(PROC, exist_ok=True)

def raw_path(uri): return f"{RAW}/{uri}.png"
def proc_path(uri): return f"{PROC}/{uri}.png"

