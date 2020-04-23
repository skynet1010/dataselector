import sys
from utils.postgres_functions import table_row_sql, insert_row

def init_analysis_params(args, conn, cur, task):
    retrain = False

    niteration = nepoch = 1
    best_acc = 0.0
    best_loss = best_exec_time = sys.float_info.max
    cur.execute(table_row_sql(args.best_results_table_name, args, task))
    res = cur.fetchall()
    if res == []:
        cur.execute(insert_row(args.best_results_table_name, args, task, niteration, nepoch, best_acc, best_loss, best_exec_time))
        conn.commit()
    else:
        _, _, best_acc, best_loss, best_exec_time = res[0]

    cur.execute(table_row_sql(args.task_states_table_name, args, task))
    res = cur.fetchall()
    if res != []:
        niteration, nepoch, _, _ = res[0]
        nepoch+=1
        retrain=True

    return retrain, niteration, nepoch, best_acc, best_loss, best_exec_time