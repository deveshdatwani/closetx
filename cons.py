import pika
import time

# Connect to RabbitMQ server
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

# Declare the queue (must match the one used by the producer)
channel.queue_declare(queue='test_queue', durable=True)

# Callback function to process the received message
def callback(ch, method, properties, body):
    print(f"Received: {body.decode()}")
    # Simulating task processing
    # time.sleep(1)
    print("Task done!")
    ch.basic_ack(delivery_tag=method.delivery_tag)  # Acknowledge message processing

# Set up the consumer to listen to the queue
channel.basic_consume(
    queue='test_queue',
    on_message_callback=callback,
    auto_ack=False  # Do not automatically acknowledge the message, wait until it's processed
)

print('Waiting for messages. To exit press CTRL+C')
channel.start_consuming()