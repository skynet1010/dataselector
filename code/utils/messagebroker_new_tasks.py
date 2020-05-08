"""
@brief: fill queue with new tasks read from tasks file.
"""

import pika
import sys
from utils.ip_provider import get_valid_ip

def create_new_tasks(fn,broker):
    tasks=[]
    with open(fn,"r") as f:
        for line in f:
            tasks.append(line)


    connection = pika.BlockingConnection(pika.ConnectionParameters(host=get_valid_ip(broker)))
    channel = connection.channel()

    channel.queue_declare(queue='task_queue', durable=True)

    for task in tasks:
        task = task.strip("\n")
        channel.basic_publish(
                exchange='',
                routing_key='task_queue',
                body=task,
                properties=pika.BasicProperties(
                    delivery_mode=2,  # make message persisten
                ))
        print(" [x] Sent %r" % task)
        
    connection.close()
