FROM python:3.10

COPY . .

RUN apt-get update && apt-get install -y libpq-dev build-essential
RUN pip install --upgrade pip
RUN apt-get install -y default-libmysqlclient-dev
RUN pip install -r requirements.txt
RUN apt-get install -y default-mysql-client

WORKDIR /app

EXPOSE 5000

CMD ["python3", "-m", "flask", "run", "--host", "0.0.0.0", "-p", "5000", "--debug"]
