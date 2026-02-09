import os, logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
RAW_DIR = os.getenv("RAW_DIR", f"{CACHE_DIR}/raw")
PROC_DIR = os.getenv("PROC_DIR", f"{CACHE_DIR}/processed")
EXT = os.getenv("IMAGE_EXT", ".png")

for d in [CACHE_DIR, RAW_DIR, PROC_DIR]:
    os.makedirs(d, exist_ok=True)
    logger.info("dir_ready", extra={"path": d})

def raw_path(uri):
    path = f"{RAW_DIR}/{uri}{EXT}"
    logger.info("raw_path", extra={"uri": uri, "path": path})
    return path

def proc_path(uri):
    path = f"{PROC_DIR}/{uri}{EXT}"
    logger.info("proc_path", extra={"uri": uri, "path": path})
    return path