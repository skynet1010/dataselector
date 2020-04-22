import pika
import psycopg2
from selector import analysis

def start_task_listener(args):

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=args.rabbitmq_server,
            heartbeat=0
        )
    )

    channel = connection.channel()

    channel.queue_declare(queue="task_queue",durable=True)

    print(" [*] Waiting for tasks. To exit press CTRL+C")

    def callback(ch,method,properties,body):
        task = body.decode("utf-8")
        print(" [x] Received " + task)
        finished_successfully = False
        try:
            conn = psycopg2.connect(host=args.database_host,database=args.database,user=args.database_user,password=args.database_password)
        except Exception as e:
            raise e
        while not finished_successfully:
            try:
                finished_successfully = analysis(conn,args,task)
                conn.close()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, halfing batch size')
                    args.batch_size=max(1,args.batch_size//2)
                    print(f"Batch size is now: {args.batch_size}")
                    continue
                else:
                    raise e
        
        print(" [x] Done |:->")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)#this is important to prefetch only one task <- heavily influence the way the tasks are spread over the cluster

    channel.basic_consume(queue="task_queue",on_message_callback=callback)

    channel.start_consuming()
