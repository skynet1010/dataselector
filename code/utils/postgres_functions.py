import sys
import time

def table_row_sql(table_name, args, task):
    return  f"""
    SELECT 
        niteration, 
        nepoch, 
        acc, 
        loss, 
        exec_time 
    FROM {table_name} 
    WHERE task='{task}';
    """ if table_name == args.best_test_results_table_name or table_name == args.best_validation_results_table_name else f"""
    WITH 
        roi AS (SELECT * FROM {table_name} WHERE task='{task}'),
        maxTimeStamp AS (SELECT MAX(timestamp) FROM roi)
    SELECT 
        roi.niteration, 
        roi.nepoch,
        roi.acc_valid,
        roi.loss_valid
    FROM 
        roi,
        maxTimeStamp
    WHERE timestamp=maxTimeStamp.max;
    """

def create_table_sql(table_name, args):
    query = ""
    if table_name == args.best_test_results_table_name or table_name == args.best_validation_results_table_name:
        query = f"""
        CREATE TABLE {table_name}(
            timestamp float PRIMARY KEY,
            task text NOT NULL,
            niteration INT NOT NULL,
            nepoch INT NOT NULL,
            acc float8 NOT NULL,
            loss float8 NOT NULL,
            exec_time float8 NOT NULL
        );
        """ 
    elif table_name == args.states_current_task_table_name: 
        query=f"""
        CREATE TABLE {table_name}(
            timestamp float PRIMARY KEY,
            task text NOT NULL,
            niteration INT NOT NULL,
            nepoch INT NOT NULL,
            acc_valid float8 NOT NULL,
            acc_train float8 NOT NULL,
            loss_valid float8 NOT NULL,
            loss_train float8 NOT NULL
        );
        """
    return query

def insert_row(table_name, args, task, niteration=0, nepoch=0, best_acc=0.0, best_loss=sys.float_info.max, best_exec_time=sys.float_info.max, curr_acc_valid = 0.0, curr_acc_train = 0.0, curr_loss_valid = sys.float_info.max, curr_loss_train = sys.float_info.max,timestamp=time.time()):
    return f"""
    INSERT INTO {table_name}(
        timestamp,task, niteration, nepoch, acc, loss, exec_time
    )
    VALUES(
         {timestamp},'{task}',{niteration},{nepoch},{best_acc},{best_loss},{best_exec_time}
    );
    """ if table_name == args.best_test_results_table_name or table_name == args.best_validation_results_table_name else f"""
    INSERT INTO {table_name}(
        timestamp,task, niteration, nepoch, acc_valid, acc_train, loss_valid, loss_train
    )
    VALUES(
        {timestamp}, '{task}',{niteration},{nepoch},{curr_acc_valid},{curr_acc_train},{curr_loss_valid},{curr_loss_train}
    );
    """

def update_row(table_name, task, niteration, nepoch, best_acc, best_loss, best_exec_time):
    return f"""
    UPDATE {table_name}
    SET 
        niteration={niteration},
        nepoch={nepoch},
        acc={best_acc},
        loss={best_loss},
        exec_time={best_exec_time}
    WHERE
        task='{task}'    
    """

def make_sure_table_exist(args, conn, cur, table_name):
    cur.execute("select exists(select * from information_schema.tables where table_name=%s)", (table_name,))
    table_exists = cur.fetchone()[0]
    if not table_exists:
        psql = create_table_sql(table_name,args)
        cur.execute(psql)
        conn.commit()