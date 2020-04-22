import sys
import time

def table_row_sql(table_name,best_results_table_name, task):
    return  f"""
    SELECT 
        niteration, 
        nepoch, 
        best_acc, 
        best_loss, 
        best_exec_time 
    FROM {table_name} 
    WHERE task='{task}';
    """ if table_name == best_results_table_name else f"""
    WITH 
        roi AS (SELECT * FROM {table_name} WHERE task='{task}'),
        maxTimeStamp AS (SELECT MAX(timestamp) FROM roi)
    SELECT 
        roi.niteration, 
        roi.nepoch,
        roi.acc_test,
        roi.loss_test
    FROM 
        roi,
        maxTimeStamp
    WHERE timestamp=maxTimeStamp.max;
    """

def create_table_sql(table_name,best_results_table_name):
    return f"""
    CREATE TABLE {table_name}(
        task text PRIMARY KEY,
        niteration INT NOT NULL,
        nepoch INT NOT NULL,
        best_acc float8 NOT NULL,
        best_loss float8 NOT NULL,
        best_exec_time float8 NOT NULL
    );
    """ if table_name == best_results_table_name else f"""
    CREATE TABLE {table_name}(
        timestamp float PRIMARY KEY,
        task text NOT NULL,
        niteration INT NOT NULL,
        nepoch INT NOT NULL,
        acc_test float8 NOT NULL,
        acc_train float8 NOT NULL,
        loss_test float8 NOT NULL,
        loss_train float8 NOT NULL
    );
    """

def insert_row(table_name, best_results_table_name, task, niteration, nepoch, best_acc=0.0, best_loss=sys.float_info.max, best_exec_time=sys.float_info.max, curr_acc_test = 0.0, curr_acc_train = 0.0, curr_loss_test = sys.float_info.max, curr_loss_train = sys.float_info.max,timestamp=time.time()):
    return f"""
    INSERT INTO {table_name}(
        task, niteration, nepoch, best_acc, best_loss, best_exec_time
    )
    VALUES(
        '{task}',{niteration},{nepoch},{best_acc},{best_loss},{best_exec_time}
    );
    """ if table_name==best_results_table_name else f"""
    INSERT INTO {table_name}(
        timestamp,task, niteration, nepoch, acc_test, acc_train, loss_test, loss_train
    )
    VALUES(
        {timestamp}, '{task}',{niteration},{nepoch},{curr_acc_test},{curr_acc_train},{curr_loss_test},{curr_loss_train}
    );
    """

def update_row(table_name, task, niteration, nepoch, best_acc, best_loss, best_exec_time):
    return f"""
    UPDATE {table_name}
    SET 
        niteration={niteration},
        nepoch={nepoch},
        best_acc={best_acc},
        best_loss={best_loss},
        best_exec_time={best_exec_time}
    WHERE
        task='{task}'    
    """

def make_sure_table_exist(args, conn, cur, table_name):
    cur.execute("select exists(select * from information_schema.tables where table_name=%s)", (table_name,))
    table_exists = cur.fetchone()[0]
    if not table_exists:
        psql = create_table_sql(table_name, args.ds_results_table_name)
        cur.execute(psql)
        conn.commit()