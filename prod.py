import pika
from time import sleep

# Connect to RabbitMQ server
connection = pika.BlockingConnection(pika.ConnectionParameters(
                                                                host='localhost',
                                                                port=5672))
channel = connection.channel()

# Declare a queue (it will be created if it doesn't exist)
channel.queue_declare(queue='test_queue', durable=True)

# Message to be sent
message = "Hello, RabbitMQ!"

# Send message to the queue
while True:
    channel.basic_publish(
        exchange='',
        routing_key='test_queue',  # queue name
        body=message,
        properties=pika.BasicProperties(
            delivery_mode=2,  # make message persistent
        )
    )
    print(f"Sent: {message}")
    sleep(2)

connection.close()