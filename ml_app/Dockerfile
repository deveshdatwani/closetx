FROM python:3.10

WORKDIR /ml-app

COPY . .

RUN apt-get update && apt-get install -y libpq-dev build-essential
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 5001

CMD ["python3", "-u", "-m", "flask", "run", "--host", "0.0.0.0", "-p", "5001", "--debug"]
