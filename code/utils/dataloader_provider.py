import torch
import os
import shutil
import multiprocessing
from hdf5_reader import Hdf5Dataset as Dataset
from consts import time_stamp



def get_results_dir(args):
    
    return os.path.join(args.results_dir,f"runs_{time_stamp}")

def get_dataloaders(args,ss, data_composition_key):

    results_dir = get_results_dir(args)

    input_filename = f"train_test_data_{ss}_supervised.hdf5"

    real_data_path = os.path.join("..","data")
    if not os.path.isdir(real_data_path):
        try:
            os.mkdir(real_data_path)
        except Exception as e:
            print(e)
            exit(1)
    full_real_input_filename = os.path.join(real_data_path,input_filename)
    if not os.path.isfile(full_real_input_filename):
        try:
            shutil.copy(os.path.join(args.data_dir,"{}".format(input_filename)),full_real_input_filename)
        except Exception as e:
            print(e)
            exit(1)

    train_ds = Dataset(full_real_input_filename, "supervised","train",data_composition_key)
    test_ds = Dataset(full_real_input_filename, "supervised","test",data_composition_key)
    cpu_count = multiprocessing.cpu_count()
    train_data_loader = torch.utils.data.DataLoader(train_ds,num_workers=cpu_count,batch_size=args.batch_size,pin_memory=True,shuffle=False)
    test_data_loader = torch.utils.data.DataLoader(test_ds,num_workers=cpu_count,batch_size=args.batch_size,pin_memory=True,shuffle=False)

    return train_data_loader, test_data_loader