FROM python:3.10

COPY . ./app

WORKDIR /app

EXPOSE 5432

RUN apt update
RUN apt upgrade

CMD ["python3", "-u", "db.py"]