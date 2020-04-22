import sys
from postgres_functions import table_row_sql, insert_row


def init_analysis_params(args, conn, cur, task):
    retrain = False

    niteration = nepoch = 1
    best_acc = 0.0
    best_loss = best_exec_time = sys.float_info.max
    cur.execute(table_row_sql(args.ds_results_table_name, args.ds_results_table_name, task))
    res = cur.fetchall()
    if res == []:
        cur.execute(insert_row(args.ds_results_table_name, args.ds_results_table_name, task, niteration, nepoch, best_acc, best_loss, best_exec_time))
        conn.commit()
    else:
        _, _, best_acc, best_loss, best_exec_time = res[0]
        nepoch+=1
        retrain=True

    if retrain:
        cur.execute(table_row_sql(args.state_table_name, args.ds_results_table_name, task))
        res = cur.fetchall()
        if res != []:
            niteration, nepoch, _, _ = res[0]
            nepoch+=1
    return retrain, niteration, nepoch, best_acc, best_loss, best_exec_time