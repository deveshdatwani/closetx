FROM python:3.10

WORKDIR /app

COPY . .

EXPOSE 5432

RUN apt update
RUN apt upgrade
RUN apt install mysql-server-core

CMD ["systemctl", "start", "mysql"]