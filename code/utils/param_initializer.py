import sys
from code.utils.postgres_functions import table_row_sql, insert_row
import time

def init_analysis_params(args, conn, cur, task):
    retrain = False

    niteration = nepoch = 1
 
    m = {
        "acc":0.0, 
        "loss":sys.float_info.max,
        "TP":0,
        "FN":0,
        "FP":0,
        "TN":0,
        "AVG_PC":0.0,
        "AVG_NC":0.0,
        "sensitivity":0.0,
        "miss_rate":0.0,
        "specificity":0.0,
        "fallout":0.0,
        "precision":0.0,
        "NPV":0.0,
        "F1":0.0,
        "exec_time":sys.float_info.max
    }

    cur.execute(table_row_sql(args.best_validation_results_table_name, args, task))
    res = cur.fetchall()
    if res == []:
        cur.execute(insert_row(args.best_validation_results_table_name, args, task, niteration, nepoch,timestamp=time.time(),m1=m))
        conn.commit()
    else:
        _, _, best_acc, best_loss, best_exec_time = res[0]

    cur.execute(table_row_sql(args.states_current_task_table_name, args, task))
    res = cur.fetchall()
    if res != []:
        niteration, nepoch, _, _ = res[0]
        nepoch+=1
        retrain=True

    return retrain, niteration, nepoch, best_acc, best_loss, best_exec_time