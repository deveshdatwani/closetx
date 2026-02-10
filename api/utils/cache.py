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

def raw_path_with_ext(uri, ext=None):
    if ext:
        if not ext.startswith('.'):
            ext = '.' + ext
    else:
        ext = EXT
    path = f"{RAW_DIR}/{uri}{ext}"
    logger.info("raw_path_with_ext", extra={"uri": uri, "path": path})
    return path

def save_raw_bytes(uri, data: bytes, ext: str = None):
    path = raw_path_with_ext(uri, ext)
    try:
        with open(path, 'wb') as f:
            f.write(data)
        logger.info("raw_saved", extra={"uri": uri, "path": path})
        return path
    except Exception:
        logger.exception("raw_save_failed", extra={"uri": uri, "path": path})
        raise

_EXTERNAL_MAP = f"{CACHE_DIR}/external_map.json"
def record_source_mapping(uri, source_url: str):
    import json
    mapping = {}
    try:
        if os.path.exists(_EXTERNAL_MAP):
            with open(_EXTERNAL_MAP, 'r') as f:
                mapping = json.load(f)
    except Exception:
        mapping = {}
    mapping[uri] = source_url
    with open(_EXTERNAL_MAP, 'w') as f:
        json.dump(mapping, f)
    logger.info("external_mapping_recorded", extra={"uri": uri, "source": source_url})
    return _EXTERNAL_MAP