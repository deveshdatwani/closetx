import boto3, os

s3 = boto3.client("s3")
BUCKET = os.getenv("S3_BUCKET")

def upload(local, key):
    s3.upload_file(local, BUCKET, key)

def delete(key):
    s3.delete_object(Bucket=BUCKET, Key=key)

