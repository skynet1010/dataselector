import sys
from utils.postgres_functions import table_row_sql, insert_row
from utils.consts import ds_results_table_name, state_table_name

def init_analysis_params(args, conn, cur, task):
    retrain = False

    niteration = nepoch = 1
    best_acc = 0.0
    best_loss = best_exec_time = sys.float_info.max
    cur.execute(table_row_sql(ds_results_table_name, task))
    res = cur.fetchall()
    if res == []:
        cur.execute(insert_row(ds_results_table_name, task, niteration, nepoch, best_acc, best_loss, best_exec_time))
        conn.commit()
    else:
        _, _, best_acc, best_loss, best_exec_time = res[0]

    cur.execute(table_row_sql(state_table_name, task))
    res = cur.fetchall()
    if res != []:
        niteration, nepoch, _, _ = res[0]
        nepoch+=1
        retrain=True
        
    return retrain, niteration, nepoch, best_acc, best_loss, best_exec_time