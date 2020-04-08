from utils.hdf5_reader import Hdf5Dataset as Dataset
import os
import torch
import numpy as np
import torch.multiprocessing as mp
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.models as models
from argparse import ArgumentParser
import time
import sys
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import socket
from math import isnan

learning_rate = 1e-3
nr_of_classes = 2

model_dict = \
{
    # "resnet18":models.resnet18,
    # "resnet50":models.resnet50,
    # "resnet101":models.resnet101,
    # "alexnet":models.alexnet,
    # "vgg16":models.vgg16,
    ##"densnet":models.densenet161,
    ##"inception":models.inception_v3,
    ##"googlenet":models.googlenet,
    "shufflenet":models.shufflenet_v2_x1_0,
    "mobilenet":models.mobilenet_v2,
    "resnext50_32x4d":models.resnext50_32x4d,
    "resnext101_32x8d":models.resnext101_32x8d,
    "wide_resnet50_2":models.wide_resnet50_2,
}

data_compositions = {
    "RGB":3,
    "NIR":1,
    "SLOPE":1,
    "ROUGHNESS":1,
    "NDVI":1,
    "DOM":1,
    "RGB_NIR":4,
    "RGB_SLOPE":4,
    "RGB_NDVI":4,
    "NIR_SLOPE":2,
    "NDVI_SLOPE":2,
    "NDVI_NIR":2,
    "RGB_NIR_SLOPE":5,
    "NDVI_NIR_SLOPE":3,
    "RGB_NIR_NDVI_SLOPE":6,
}


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def train(model, train_data_loader, criterion, optimizer,batch_size):
    running_loss = 0
    correct= 0
    total=0
    for step,data in enumerate(train_data_loader):
        tmp_batch_size = len(data["labels"])
        lbl_onehot = torch.FloatTensor(tmp_batch_size,nr_of_classes).cuda()

        # =============datapreprocessing=================
        img = torch.FloatTensor(data["imagery"].float()).cuda()
        lbl_onehot.zero_()
        lbl_onehot = lbl_onehot.scatter(1,data["labels"].cuda(),1).cuda()
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, lbl_onehot)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        #determine acc
        _, predicted = torch.max(output.data, 1)
        total += tmp_batch_size
        correct += (predicted.cpu() == data["labels"].view(tmp_batch_size)).sum().item()
    return running_loss, correct/total

def test(model, test_data_loader,criterion,optimizer,batch_size):
    correct = 0
    total = 0
    running_loss=0
    with torch.no_grad():
        for data in test_data_loader:
            tmp_batch_size = len(data["labels"])
            lbl_onehot = torch.FloatTensor(tmp_batch_size,nr_of_classes).cuda()
            # =============datapreprocessing=================
            img = torch.FloatTensor(data["imagery"].float()).cuda()
            # ===================forward=====================
            output = model(img)
            
            lbl_onehot.zero_()
        
            lbl_onehot = lbl_onehot.scatter(1,data["labels"].cuda(),1).cuda()
            loss = criterion(output, lbl_onehot)
            running_loss+=loss.item()
            _, predicted = torch.max(output.data, 1)
            total += tmp_batch_size
            correct += (predicted.cpu() == data["labels"].view(tmp_batch_size)).sum().item()
    return running_loss,correct,total

def manipulateModel(model_name, is_feature_extraction,dim):
    model = model_dict[model_name](pretrained=is_feature_extraction)
    set_parameter_requires_grad(model, True)
    #output layer
    if model_name == "resnet18" or \
        model_name == "resnet50" or \
        model_name == "resnet101" or \
        model_name == "inception" or \
        model_name == "googlenet" or \
        model_name == "shufflenet" or \
        model_name == "resnext50_32x4d" or \
        model_name == "resnext101_32x8d" or \
        model_name == "wide_resnet50_2":
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, nr_of_classes)
    elif model_name =="alexnet" or model_name == "vgg16" or model_name=="mobilenet":
        layer_number = 0
        if model_name == "alexnet" or model_name == "vgg16":
            layer_number = 6
        elif model_name == "mobilenet":
            layer_number = 1
        num_ftrs = model.classifier[layer_number].in_features
        model.classifier[layer_number] = nn.Linear(num_ftrs,nr_of_classes)
    elif model_name == "densnet":
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs,nr_of_classes)    
    
    #input layer:
    if dim!=3:
        if model_name =="alexnet" or model_name =="vgg16":
            layer_index = 0
            model.features[layer_index] = nn.Conv2d(dim,64,kernel_size=(11,11),stride=(4,4),padding=(2,2))
        elif model_name == "resnet18" or model_name=="resnet50" or model_name=="resnet101" or model_name == "googlenet" or model_name=="resnext50_32x4d" or model_name=="resnext101_32x8d" or model_name == "wide_resnet50_2":
            model.conv1 = nn.Conv2d(dim,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        elif model_name == "densnet":
            model.features.conv0 = nn.Conv2d(dim,96,kernel_size=(7,7),stride=(2,2),padding=(3,3), bias=False)
        elif model_name == "inception":
            model.Conv2d_1a_3x3.conv = nn.Conv2d(dim,32,kernel_size=(3,3),stride=(2,2),bias=False)
        elif model_name =="shufflenet":
            model.conv1[0] = nn.Conv2d(dim,24,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        elif model_name == "mobilenet":
            model.features[0][0] = nn.Conv2d(dim,32,kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)



    return model.cuda()

def print_all_model_architectures(is_feature_extraction,dim):
    with open("arch.txt","w") as f:
        sys.stdout = f
        for key in model_dict.keys():
            print(key)
            model = manipulateModel(key,is_feature_extraction,dim)
            print(model)
            print("########################################################################################################")
            print("########################################################################################################")
            print("########################################################################################################")

def get_device_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    return s.getsockname()[0].split(".")[-1]

def create_task_pool(task_pool_fn):
    with open(task_pool_fn,"w") as f:
        sys.stdout = f
        print(",".join(data_compositions.keys()))
        
def analysis(input_filename,key, ss, val_dim, args, df, metadata_fn, shared_dir, device_address,prev_acc,exec_time_best_config,start_iteration=0,start_epoch=0,data_key="RGB"):

    train_ds = Dataset(os.path.join(shared_dir,"{}".format(input_filename)), "supervised","train",data_key)
    test_ds = Dataset(os.path.join(shared_dir,"{}".format(input_filename)),  "supervised","test",data_key)


    train_data_loader = torch.utils.data.DataLoader(train_ds,num_workers=args.workers,batch_size=args.batch_size,pin_memory=True,shuffle=False)
    test_data_loader = torch.utils.data.DataLoader(test_ds,num_workers=args.workers,batch_size=args.batch_size,pin_memory=True,shuffle=False)

    s = []
    prev_loss = sys.maxsize
    best_acc_iterations = 0
    best_loss_iterations = sys.maxsize

    for iteration_number in range(start_iteration,args.iterations):
        writer = SummaryWriter(log_dir="best_model_search_runs",comment=f"key={key} search_size={ss} iteration={iteration_number}")

        print(key,iteration_number)
        df.at[iteration_number,key] = [sys.maxsize,0.0,sys.maxsize,0.0]
        
        model = manipulateModel(key,args.is_feature_extraction,val_dim)
        if iteration_number == 0:
            data = next(iter(train_data_loader))
            writer.add_graph(model,torch.FloatTensor(data["imagery"].float()).cuda())


        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

        for epoch in range(start_epoch,args.epochs):
            try:
                start = time.time()
                loss_train,acc_train = train(model,train_data_loader,criterion,optimizer,args.batch_size) 
                loss_test,correct, total =  test(model,test_data_loader,criterion,optimizer,args.batch_size)
                exec_time_curr = time.time()-start
                acc_test = correct/total
                s.append([loss_train,loss_test,acc_train,acc_test])
                writer.add_scalars('{}/Loss'.format(key), {"train":loss_train,"test":loss_test}, epoch)
                writer.add_scalars('{}/Accuracy'.format(key), {"train":acc_train,"test":acc_test}, epoch)
                if acc_test > best_acc_iterations:
                    best_acc_iterations = acc_test
                    best_loss_iterations = loss_test
                elif acc_test == best_acc_iterations and best_loss_iterations < loss_test:
                    best_loss_iterations = loss_test
                print('epoch [{}/{}], loss:{:.4f}, acc {}/{} = {:.4f}%, time: {}'.format(epoch+1, args.epochs, loss_test, correct,total,acc_test*100, exec_time_curr))
                if acc_test > prev_acc:
                    torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict()}, './best_input_alexnet_{}.pth'.format(device_address))
                    with open(os.path.join(shared_dir,"best_in_composition_{}.txt".format(device_address)),"w") as f:
                        f.write(key)
                    prev_acc = acc_test
                    prev_loss = loss_test
                    exec_time_best_config = exec_time_curr
                elif (acc_test == prev_acc and exec_time_best_config > exec_time_curr) or (acc_test == prev_acc and loss_test < prev_loss):
                    torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict()}, './best_input_alexnet_{}.pth'.format(device_address))
                    with open(os.path.join(shared_dir,"best_in_composition_{}.txt".format(device_address)),"w") as f:
                        f.write(key)
                    exec_time_best_config = exec_time_curr
                    prev_loss = loss_test
                if acc_test > df.at[iteration_number,key][3]:
                    df.at[iteration_number,key] = [loss_train,acc_train,loss_test,acc_test]
            except KeyboardInterrupt as e:
                print("Handling KeyboardInterrupt \n Saving model ...")
                df.to_csv(metadata_fn.split(".")[0]+"_keyboard_interrupt_{}.csv".format(device_address))       
                torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict()}, './best_input_alexnet_keyboard_interrupt_{}.pth'.format(device_address))
                print("Model saved! Thanks for waiting :)")
                print("SAVE current state!...")
                with open(os.path.join(shared_dir, "current_state_{}.txt".format(device_address)),"w")as f:
                    f.write("{},{},{},{},{},{},{}".format(key,iteration_number,epoch,args.tp,exec_time_best_config,prev_acc,prev_loss))
                print("Current state saved")
                print(e)
                print("GOODBY :)))")
                exit(0)
            
            
        df.to_csv(metadata_fn)       
    writer.add_hparams({"key":key,"ss":ss},{"hparam/accuracy":best_acc_iterations,"hparam/loss":best_loss_iterations})
    return prev_acc,exec_time_best_config

def _create_shared_dir(args,search_size):
    shared_dir = os.path.join(args.hdir,args.u,args.sdir,search_size) if args.is_docker==0 else os.path.join("/","workspace","dev","data","{}".format(search_size))  
    if not os.path.isdir(shared_dir):
        os.mkdir(shared_dir)
    return shared_dir

def _create_meta_data_df(shared_dir,device_address):
    metadata_fn = os.path.join(shared_dir,"meta_data_best_model_{}.csv".format(device_address))
    df = pd.DataFrame(columns=model_dict.keys()) if not os.path.isfile(metadata_fn) else pd.read_csv(metadata_fn)
    if os.path.isfile(metadata_fn):
        df = pd.read_csv(metadata_fn)
        for key in list(df)[1:]:
            df.loc[df[key].isnull(),key] =df.loc[df[key].isnull(),key].apply(lambda x: [np.nan,np.nan,np.nan,np.nan])
    return df, metadata_fn

def _get_prev_config(shared_dir,device_address):
    of = os.path.join(shared_dir, "current_state_{}.txt".format(device_address))
    prev_run_config = None
    if os.path.isfile(of):
        with open(of,"r") as f:
            prev_run_config = f.readline().split(",")
        os.remove(of)
    return prev_run_config

def main():
    parser = ArgumentParser()
    parser.add_argument("-bs", "--batch_size",dest="batch_size", default=512,type=int)
    parser.add_argument("-e","--epochs", dest="epochs", default=10,type=int)
    parser.add_argument("-i","--iterations", dest="iterations", default=3,type=int)
    parser.add_argument("-isd","--is_docker", dest="is_docker", default=1,type=int)
    parser.add_argument("-ife", "--is_feature_extraction", dest="is_feature_extraction", default=1,type=int,help="0=False,1=True")
    parser.add_argument("-sdir","--shared_dir", dest="sdir", default="halloffame_analysis")
    parser.add_argument("-u","--user", dest="u", default="user")
    parser.add_argument("-tp", "--task-pool",dest="tp", default=0,type=int,help="0=False,1=True")
    parser.add_argument("-hdir","--home_dir", dest="hdir", default="/home")
    parser.add_argument("-k","--keys",dest="keys",default="RGB,RGB,RGB,RGB")
    parser.add_argument("-w","--workers", dest="workers", default=16,type=int,help="Determines the number of worker threads for reading data.")
    
    args = parser.parse_args()

    keys = args.keys.split(",")

    for idx,ss in enumerate([f"ss{8*i}"for i in range(4,5)]):
        input_filename = f"train_test_data_{ss}_supervised.hdf5"
        device_address = get_device_address()

        shared_dir = _create_shared_dir(args,ss)

        df,metadata_fn = _create_meta_data_df(shared_dir,device_address)
        #TODO save more information about sigint

        prev_acc = 0
        exec_time_best_config = sys.maxsize

        prev_run_config = _get_prev_config(shared_dir,device_address)

        start_iteration = start_epoch = 0

        task=""
        
        old_batch_size = args.batch_size
        if args.tp == 0:
            if prev_run_config!=None:
                task = prev_run_config[0]
                start_iteration = int(prev_run_config[1])
                start_epoch = int(prev_run_config[2])
                args.tp = int(prev_run_config[3])
                exec_time_best_config = float(prev_run_config[4])
                prev_acc = float(prev_run_config[5])
                prev_run_config=None
            for key in model_dict.keys():
                if task != "" and key!=task:
                    continue
                else:
                    task=""
                args.batch_size = old_batch_size
                data_key = keys[idx]
                val_dim = data_compositions[data_key]
                not_finished_successfully = True
                while not_finished_successfully:
                    try:
                        prev_acc,exec_time_best_config = analysis(input_filename,key,ss,val_dim, args, df, metadata_fn, shared_dir, device_address,prev_acc,exec_time_best_config,start_iteration,start_epoch,data_key)
                        not_finished_successfully = False
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print('| WARNING: ran out of memory, halfing batch size')
                            args.batch_size=max(1,args.batch_size//2)
                            print(f"Batch size is now: {args.batch_size}")
                            continue
                        else:
                            raise e


        else: 
            tp_fn = os.path.join(args.hdir,args.u,args.sdir,ss,"task_pool.txt")
            while True:
                task = ""
                splitted_line = []
                if prev_run_config==None:
                    with open(tp_fn, "r") as f:
                        splitted_line = [split_item.rstrip("\n") for split_item in f.readline().split(",")]
                        time.sleep(1)
                    with open(tp_fn,"w")as f:
                        if splitted_line[0]!="":
                            task = splitted_line.pop(0)
                            f.write(",".join(splitted_line))
                        else:
                            break
                else:
                    task = prev_run_config[0]
                    start_iteration = int(prev_run_config[1])
                    start_epoch = int(prev_run_config[2])
                    args.tp = int(prev_run_config[3])
                    exec_time_best_config = float(prev_run_config[4])
                    prev_acc = float(prev_run_config[5])
                    prev_run_config=None
                    
                prev_acc,exec_time_best_config=analysis(input_filename,task,ss,data_compositions[task],args,df, metadata_fn, shared_dir, device_address,prev_acc,exec_time_best_config,start_iteration, start_epoch)

if __name__=="__main__":
    main()