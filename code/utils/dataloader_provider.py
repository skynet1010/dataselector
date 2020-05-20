import torch
import os
import shutil
import multiprocessing
from code.utils.hdf5_reader import Hdf5Dataset as Dataset
from code.utils.consts import time_stamp
from torch.utils.data.sampler import SubsetRandomSampler
from numpy import floor

def get_results_dir(args):
    
    return os.path.join(args.results_dir,args.run_dir)

def get_dataloaders(args,ss, data_composition_key,model_key,validation=True):

    results_dir = get_results_dir(args)

    input_filename = f"train_test_data_{ss}_supervised.hdf5"

    real_data_path = os.path.join("..","data")
    if not os.path.isdir(real_data_path):
        try:
            os.mkdir(real_data_path)
        except Exception as e:
            print(e)
            exit(1)

    valid_ss = ["ss8","ss16","ss24","ss32"]
    for v_ss in valid_ss:
        if v_ss!=ss:
            old_input_filename = os.path.join(real_data_path,f"train_test_data_{v_ss}_supervised.hdf5")
            if not os.path.isfile(old_input_filename):
                continue
            else:
                os.remove(old_input_filename)

        
    full_real_input_filename = os.path.join(real_data_path,input_filename)
    if not os.path.isfile(full_real_input_filename):
        try:
            shutil.copy(os.path.join(args.data_dir,"{}".format(input_filename)),full_real_input_filename)
        except Exception as e:
            print(e)
            exit(1)

    train_ds = Dataset(full_real_input_filename, "supervised","train",data_composition_key,model_key)
    test_ds = Dataset(full_real_input_filename, "supervised","test",data_composition_key, model_key)
    cpu_count = multiprocessing.cpu_count()

    test_data_loader = torch.utils.data.DataLoader(test_ds,num_workers=cpu_count,batch_size=args.batch_size,pin_memory=True,shuffle=False)

    if validation:
        validation_split = 0.142857143
        train_ds_size = len(train_ds)
        indices = list(range(train_ds_size))
        split = int(floor(validation_split * train_ds_size))

        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_data_loader = torch.utils.data.DataLoader(train_ds,num_workers=cpu_count,batch_size=args.batch_size,pin_memory=True,shuffle=False,sampler=train_sampler)
        valid_data_loader = torch.utils.data.DataLoader(train_ds,num_workers=cpu_count,batch_size=args.batch_size,pin_memory=True,shuffle=False,sampler=valid_sampler)

        return train_data_loader, valid_data_loader, test_data_loader

    train_data_loader = torch.utils.data.DataLoader(train_ds,num_workers=cpu_count,batch_size=args.batch_size,pin_memory=True,shuffle=False)

    return train_data_loader, test_data_loader