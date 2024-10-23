FROM python:3.10

COPY . .

RUN apt-get update

RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

ENV FLASK_APP=app.py
ENV FLASK_ENV=production

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0" "-p=5000"]
