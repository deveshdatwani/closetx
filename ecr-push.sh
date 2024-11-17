aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 205625473463.dkr.ecr.us-east-1.amazonaws.com

sudo docker push 205625473463.dkr.ecr.us-east-1.amazonaws.com/closetx:v1

