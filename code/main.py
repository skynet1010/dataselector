from selectBestDataComposition import data_selector
from argparse import ArgumentParser
import os

def main():
    parser = ArgumentParser()
    parser.add_argument("-k","--kind_of_analysis",dest="kind_of_analysis",default="data")
    parser.add_argument("-bs", "--batch_size",dest="batch_size", default=128,type=int)
    parser.add_argument("-e","--epochs", dest="epochs", default=20,type=int)
    parser.add_argument("-i","--iterations", dest="iterations", default=5,type=int)
    parser.add_argument("-ife", "--is_feature_extraction", dest="is_feature_extraction", default=1,type=int,help="0=False,1=True")
    parser.add_argument("-d", "--data_dir", dest="data_dir", default=os.path.join("..","shared","data"))
    parser.add_argument("-r", "--results_dir", dest="results_dir", default=os.path.join("..","shared","results"))
    parser.add_argument("-rmqs", "--rabbitmq_server", dest="rabbitmq_server", default="10.96.63.220") #sollte durch den k8s proxy aufgelöst werden.
    parser.add_argument("-dbh", "--database_host", dest="database_host", default="10.104.105.248") #sollte durch den k8s proxy aufgelöst werden.
    parser.add_argument("-db", "--database", dest="database", default="elfi")
    parser.add_argument("-dbu", "--database_user", dest="database_user", default="user")
    parser.add_argument("-dbpw", "--database_password", dest="database_password", default="password123")
    args = parser.parse_args()

    if args.kind_of_analysis == "data":
        data_selector.start_task_listener(args)
    elif args.kind_of_analysis == "model":
        #TODO muss noch näher spezifiziert werden.
        pass

    

if __name__ == "__main__":
    main()
