import pika
import psycopg2
from code.selector import analysis
from code.utils.messagebroker_new_tasks import create_new_tasks
from code.utils.consts import model_dict
from code.BOBO_hypterparameter_search import hyperparameter_optimization

def get_best_data_composition(conn,args):
    fn = "automatic_generated_tasks.txt"
    
    cur = conn.cursor()
    query = lambda x: f"WITH acc AS(SELECT MAX(acc) as max_acc FROM {args.best_test_results_table_name} WHERE task like 'ss{x}%')  SELECT task FROM {args.best_test_results_table_name}, acc WHERE acc=max_acc and task like 'ss{x}%';"
    with open(fn, "w") as f:
        for ss in [8,16,32]:
            cur.execute(query(ss))
            res = cur.fetchall()
            task=res[0][0]
            ts = task.split(":")[:2]

            for model in model_dict.keys():
                line = ":".join(ts)
                line+=(":"+model+"\n")
                f.write(line)
    create_new_tasks(fn,args.rabbitmq_server)

    return True

def hyperparameter_optimization_task_creator(conn,args):
    fn = "automatic_generated_tasks.txt"
    
    cur = conn.cursor()
    query = lambda x: f"WITH acc AS(SELECT MAX(acc) as max_acc FROM {args.best_test_results_table_name} WHERE task like 'ss{x}%')  SELECT task FROM {args.best_test_results_table_name}, acc WHERE acc=max_acc and task like 'ss{x}%';"
    with open(fn, "w") as f:
        for ss in [8,16,32]:
            cur.execute(query(ss))
            res = cur.fetchall()
            task=res[0][0]
            task= task.rstrip()
            task="HO:"+task
            f.write(task)
    create_new_tasks(fn,args.rabbitmq_server)

    return True
    
def start_task_listener(args):

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=args.rabbitmq_server,
            heartbeat=0
        )
    )

    channel = connection.channel()

    channel.queue_declare(queue="task_queue_final",durable=True)

    print(" [*] Waiting for tasks. To exit press CTRL+C")

    def callback(ch,method,properties,body):
        init_batch_size = args.batch_size
        task = body.decode("utf-8")
        print(" [x] Received " + task)
        finished_successfully = False
        try:
            conn = psycopg2.connect(host=args.database_host,database=args.database,user=args.database_user,password=args.database_password)
        except Exception as e:
            raise e
        while not finished_successfully:
            try:
                if task=="check_best_composition":
                    finished_successfully = get_best_data_composition(conn,args)
                elif task=="create_hyperparameter_optimization_task":
                    finished_successfully = hyperparameter_optimization_task_creator(conn,args)
                elif len(task.split(":"))[0] == "HO":
                    finished_successfully = hyperparameter_optimization(conn,args,task)
                else:
                    finished_successfully = analysis(conn,args,task)
                conn.close()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, halfing batch size')
                    args.batch_size=max(1,args.batch_size//2)
                    print(f"Batch size is now: {args.batch_size}")
                    continue
                else:
                    print(e)
                    exit(1) 
            except Exception as e:
                print(e)
                exit(1)
                    
        args.batch_size = init_batch_size
        print(" [x] Done |:->")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)#this is important to prefetch only one task <- heavily influence the way the tasks are spread over the cluster

    channel.basic_consume(queue="task_queue_final",on_message_callback=callback)

    channel.start_consuming()
