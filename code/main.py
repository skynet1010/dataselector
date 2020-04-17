from selectBestDataComposition import data_selector
from argparse import ArgumentParser
import os
import re 
import socket
# Make a regular expression 
# for validating an Ip-address 
regex = '''^(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.( 
            25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.( 
            25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.( 
            25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)'''
      
# Define a function for 
# validate an Ip addess 
def ip_address_is_valid(Ip):  
  
    # pass the regular expression 
    # and the string in search() method 
    if(re.search(regex, Ip)):  
        return True          
    else:  
        return False


def main():
    parser = ArgumentParser()
    parser.add_argument("-k","--kind_of_analysis",dest="kind_of_analysis",default="data")
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
    # parser.add_argument("-uin", "--state_update_intervall", dest="state_update_intervall", default=5,type=int)
    args = parser.parse_args()
    if args.kind_of_analysis == "data":
        if not ip_address_is_valid(args.rabbitmq_server):
            addr = socket.gethostbyname(args.rabbitmq_server)
            print(addr)
            args.rabbitmq_server = addr
        if not ip_address_is_valid(args.database_host):
            args.database_host = socket.gethostbyname(args.database_host)
        data_selector.start_task_listener(args)
    elif args.kind_of_analysis == "model":
        #TODO muss noch näher spezifiziert werden.
        pass

    

if __name__ == "__main__":
    main()
