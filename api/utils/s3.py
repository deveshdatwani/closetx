import boto3, os, logging

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET = os.getenv("S3_BUCKET")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)
logger.info("s3_client_initialized", extra={"bucket": BUCKET})

def upload(local, key):
    logger.info("s3_upload_start", extra={"local": local, "key": key})
    s3.upload_file(local, BUCKET, key)
    logger.info("s3_upload_done", extra={"key": key})

def delete(key):
    logger.info("s3_delete_start", extra={"key": key})
    s3.delete_object(Bucket=BUCKET, Key=key)
    logger.info("s3_delete_done", extra={"key": key})