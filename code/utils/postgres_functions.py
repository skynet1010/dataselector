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
            exec_time float8 NOT NULL,
            TP INT NOT NULL,
            FN INT NOT NULL,
            FP INT NOT NULL,
            TN INT NOT NULL,
            AVG_PC float8 NOT NULL,
            AVG_NC float8 NOT NULL,
            sensitivity float8 NOT NULL,
            miss_rate float8 NOT NULL,
            specificity float8 NOT NULL,
            fallout float8 NOT NULL,
            precision float8 NOT NULL,
            NPV float8 NOT NULL,
            F1 float8 NOT NULL
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

def insert_row(table_name, args, task, niteration=0, nepoch=0, timestamp=0, m1=None, m2=None):#m1 = valid/test m2 = train
    return f"""
    INSERT INTO {table_name}(
        timestamp,task, niteration, nepoch, acc, loss, exec_time, TP, FN, FP, TN, AVG_PC, AVG_NC, sensitivity, miss_rate, specificity, fallout, precision, NPV, F1
    )
    VALUES(
         {timestamp},'{task}',{niteration},{nepoch},{m1["acc"]},{m1["loss"]},{m1["exec_time"]},{m1["TP"]}, {m1["FN"]}, {m1["FP"]}, {m1["TN"]}, {m1["AVG_PC"]}, {m1["AVG_NC"]}, {m1["sensitivity"]}, {m1["miss_rate"]}, {m1["specificity"]}, {m1["fallout"]}, {m1["precision"]}, {m1["NPV"]}, {m1["F1"]}
    );
    """ if table_name == args.best_test_results_table_name or table_name == args.best_validation_results_table_name else f"""
    INSERT INTO {table_name}(
        timestamp,task, niteration, nepoch, acc_valid, acc_train, loss_valid, loss_train
    )
    VALUES(
        {timestamp}, '{task}',{niteration},{nepoch},{m1["acc"]},{m2["acc"]},{m1["loss"]},{m2["loss"]}
    );
    """

def update_row(table_name, task, niteration, nepoch, m):
    return f"""
    UPDATE {table_name}
    SET 
        niteration={niteration},
        nepoch={nepoch},
        acc={m["acc"]},
        loss={m["loss"]},
        exec_time={m["exec_time"]},
        TP={m["TP"]}, 
        FN={m["FN"]}, 
        FP={m["FP"]}, 
        TN={m["TN"]}, 
        AVG_PC={m["AVG_PC"]}, 
        AVG_NC={m["AVG_NC"]}, 
        sensitivity={m["sensitivity"]}, 
        miss_rate={m["miss_rate"]}, 
        specificity={m["specificity"]}, 
        fallout={m["fallout"]}, 
        precision={m["precision"]}, 
        NPV={m["NPV"]}, 
        F1={m["F1"]}
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