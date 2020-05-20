from argparse import ArgumentParser
import os
import task_listener
from code.utils.ip_provider import get_valid_ip


def main():
    parser = ArgumentParser()
    parser.add_argument("-bs", "--batch_size",dest="batch_size", default=128,type=int)
    parser.add_argument("-e","--epochs", dest="epochs", default=20,type=int)
    parser.add_argument("-i","--iterations", dest="iterations", default=5,type=int)
    parser.add_argument("-ife", "--is_feature_extraction", dest="is_feature_extraction", default=1,type=int,help="0=False,1=True")
    parser.add_argument("-d", "--data_dir", dest="data_dir", default=os.path.join("..","shared","data"))
    parser.add_argument("-r", "--results_dir", dest="results_dir", default=os.path.join("..","shared","results"))
    parser.add_argument("-rmqs", "--rabbitmq_server", dest="rabbitmq_server", default="messagebroker") #sollte durch den k8s proxy aufgelöst werden.
    parser.add_argument("-dbh", "--database_host", dest="database_host", default="postgres") #sollte durch den k8s proxy aufgelöst werden.
    parser.add_argument("-db", "--database", dest="database", default="elfi")
    parser.add_argument("-dbu", "--database_user", dest="database_user", default="user")
    parser.add_argument("-dbpw", "--database_password", dest="database_password", default="password123")
    parser.add_argument("-eit", "--earlystopping_it", dest="earlystopping_it", default=5,type=int)
    parser.add_argument("-btrtn", "--best_test_results_table_name", dest="best_test_results_table_name", default="best_test_results_final")
    parser.add_argument("-bvrtn", "--best_validation_results_table_name", dest="best_validation_results_table_name", default="best_validation_results_final")
    parser.add_argument("-scttn", "--states_current_task_table_name", dest="states_current_task_table_name", default="states_current_task_final")
    parser.add_argument("-rd", "--run_dir", dest="run_dir", default="nips2020_runs")
    args = parser.parse_args()
    
    args.rabbitmq_server = get_valid_ip(args.rabbitmq_server)
    args.database_host = get_valid_ip(args.database_host)

    task_listener.start_task_listener(args)

if __name__ == "__main__":
    main()
