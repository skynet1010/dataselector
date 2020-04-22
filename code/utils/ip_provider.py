import re 
import socket

      
# Define a function for validate an Ip addess 
def get_valid_ip(Ip):  
    # Make a regular expression for validating an Ip-address 
    regex = '''^(25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.( 
            25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.( 
            25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)\.( 
            25[0-5]|2[0-4][0-9]|[0-1]?[0-9][0-9]?)'''
  
    # pass the regular expression and the string in search() method 
    if(re.search(regex, Ip)):  
        return Ip       
    else:  
        return socket.gethostbyname(Ip)