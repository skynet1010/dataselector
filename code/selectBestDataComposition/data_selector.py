import torch
from utils.hdf5_reader import Hdf5Dataset as Dataset
import os
import numpy as np
import torch.multiprocessing as mp
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.models as models
import time
import sys
from torch.utils.tensorboard import SummaryWriter
import socket
from math import isnan
import pika
import psycopg2
import shutil

learning_rate = 1e-3
nr_of_classes = 2

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
        #print(img.size())
        lbl_onehot.zero_()
        #reshaped_tensor = torch.reshape(data["labels"],(len(data["labels"]),1))
        #print(reshaped_tensor.size())
        lbl_onehot = lbl_onehot.scatter(1,data["labels"].cuda(),1).cuda()
        #print(lbl_onehot,data["labels"])
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

def manipulateModel(is_feature_extraction,dim):
    model = models.alexnet(pretrained=is_feature_extraction)
    set_parameter_requires_grad(model, True)
    #output layer
    layer_number = 6
    num_ftrs = model.classifier[layer_number].in_features
    model.classifier[layer_number] = nn.Linear(num_ftrs,nr_of_classes)
    #input layer
    if dim!=3:
        layer_index = 0
        model.features[layer_index] = nn.Conv2d(dim,64,kernel_size=(11,11),stride=(4,4),padding=(2,2))
        
    return model.cuda()
    
def table_row_sql(table_name, task):
    return  f"""
    SELECT 
        niteration, 
        nepoch, 
        best_acc, 
        best_loss, 
        best_exec_time 
    FROM {table_name} 
    WHERE task='{task}';
    """ if table_name == "ds_best_results" else f"""
    WITH 
        roi AS (SELECT * FROM {table_name} WHERE task='{task}'),
        maxTimeStamp AS (SELECT MAX(timestamp) FROM roi)
    SELECT 
        roi.niteration, 
        roi.nepoch
    FROM 
        roi,
        maxTimeStamp
    WHERE timestamp=maxTimeStamp.max;
    """

#WITH roi AS (SELECT * FROM ds_results WHERE niteration=5), max_acc AS (SELECT MAX(best_acc) FROM roi) SELECT roi.* FROM roi, max_acc WHERE best_acc=max_acc.max;

def create_table_sql(table_name):
    return f"""
    CREATE TABLE {table_name}(
        task text PRIMARY KEY,
        niteration INT NOT NULL,
        nepoch INT NOT NULL,
        best_acc float8 NOT NULL,
        best_loss float8 NOT NULL,
        best_exec_time float8 NOT NULL
    );
    """ if table_name == "ds_best_results" else f"""
    CREATE TABLE {table_name}(
        timestamp float PRIMARY KEY,
        task text NOT NULL,
        niteration INT NOT NULL,
        nepoch INT NOT NULL,
        acc_test float8 NOT NULL,
        acc_train float8 NOT NULL,
        loss_test float8 NOT NULL,
        loss_train float8 NOT NULL,
    );
    """

def insert_row(table_name, task, niteration, nepoch, best_acc=0.0, best_loss=sys.float_info.max, best_exec_time=sys.float_info.max, curr_acc_test = 0.0, curr_acc_train = 0.0, curr_loss_test = sys.float_info.max, curr_loss_train = sys.float_info.max,timestamp=time.time()):
    return f"""
    INSERT INTO {table_name}(
        task, niteration, nepoch, best_acc, best_loss, best_exec_time
    )
    VALUES(
        '{task}',{niteration},{nepoch},{best_acc},{best_loss},{best_exec_time}
    );
    """ if table_name=="ds_best_results" else f"""
    INSERT INTO {table_name}(
        timestamp,task, niteration, nepoch, acc_test, acc_train, loss_test, loss_train
    )
    VALUES(
        {timestamp}, '{task}',{niteration},{nepoch},{curr_acc_test},{curr_acc_train},{curr_loss_test},{curr_loss_train}
    );
    """

def update_row(table_name, task, niteration, nepoch, best_acc, best_loss, best_exec_time):
    return f"""
    UPDATE {table_name}
    SET 
        niteration={niteration},
        nepoch={nepoch},
        best_acc={best_acc},
        best_loss={best_loss},
        best_exec_time={best_exec_time}
    WHERE
        task='{task}'    
    """

def analysis(conn,args,task):
    
    cur = conn.cursor()

    ss,data_composition_key=task.split(":")

    results_dir = os.path.join(args.results_dir,"runs")

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

    train_data_loader = torch.utils.data.DataLoader(train_ds,num_workers=16,batch_size=args.batch_size,pin_memory=True,shuffle=False)
    test_data_loader = torch.utils.data.DataLoader(test_ds,num_workers=16,batch_size=args.batch_size,pin_memory=True,shuffle=False)

    state_table_name = "task_states"
    ds_results_table_name = "ds_best_results"

    def make_sure_table_exist(table_name):
        cur.execute("select exists(select * from information_schema.tables where table_name=%s)", (table_name,))
        table_exists = cur.fetchone()[0]
        if not table_exists:
            psql = create_table_sql(table_name)
            cur.execute(psql)
            conn.commit()

    make_sure_table_exist(state_table_name)
    make_sure_table_exist(ds_results_table_name)

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
        nepoch+=1
        retrain=True

    if retrain:
        cur.execute(table_row_sql(state_table_name, task))
        res = cur.fetchall()
        if res != []:
            niteration, nepoch = res[0]
            nepoch+=1
    

    for iteration in range(niteration,args.iterations+1):
        best_checkpoint_path = os.path.join(results_dir,f"{data_composition_key}",f"{ss}","best_alexnet.pth")

        state_checkpoint_path = os.path.join(results_dir,f"{data_composition_key}",f"{ss}","state_alexnet.pth")
        #writer = SummaryWriter(log_dir=os.path.join(results_dir,f"{data_composition_key}",f"{ss}",f"iteration_{iteration}"))
        print(data_composition_key,iteration)
        
        model = manipulateModel(args.is_feature_extraction,data_compositions[data_composition_key])

        # if iteration == 1:
        #     data = next(iter(train_data_loader))
        #     writer.add_graph(model,torch.FloatTensor(data["imagery"].float()).cuda())

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

        if retrain:
            if os.path.isfile(state_checkpoint_path):
                state_checkpoint = torch.load(state_checkpoint_path)
                model.load_state_dict(state_checkpoint["model_state_dict"])
                optimizer.load_state_dict(state_checkpoint["optimizer_state_dict"])
                print("Model loaded from file.")
            retrain=False

        update = False
        no_improve_it = 0

        best_acc_curr_iteration = 0.0
        best_loss_curr_iteration = sys.float_info.max
        for epoch in range(nepoch,args.epochs+1):
            try:
                start = time.time()
                loss_train,acc_train = train(model,train_data_loader,criterion,optimizer,args.batch_size) 
                loss_test,correct, total =  test(model,test_data_loader,criterion,optimizer,args.batch_size)
                curr_exec_time = time.time()-start
                acc_test = correct/total
                # writer.add_scalars('{}/Loss'.format(data_composition_key), {"train":loss_train,"test":loss_test}, epoch)
                # writer.add_scalars('{}/Accuracy'.format(data_composition_key), {"train":acc_train,"test":acc_test}, epoch)
                if acc_test > best_acc:
                    best_acc = acc_test
                    best_loss = loss_test
                    update=True
                elif acc_test == best_acc and best_loss < loss_test:
                    best_loss_iterations = loss_test
                    update=True
                elif acc_test == best_acc and best_loss == loss_test:
                    update=True
                if update:
                    best_acc_curr_iteration = acc_test
                    best_loss_curr_iteration = loss_test
                    no_improve_it = 0
                    best_exec_time = curr_exec_time
                    torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict()}, best_checkpoint_path)
                    cur.execute(update_row(ds_results_table_name,task,iteration,epoch,best_acc,best_loss,best_exec_time))
                    conn.commit()
                    update=False
                elif acc_test > best_acc_curr_iteration or loss_test < best_loss_curr_iteration:
                    best_acc_curr_iteration = acc_test
                    best_loss_curr_iteration = loss_test
                    no_improve_it = 0
                else:
                    no_improve_it+=1
                torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict()}, state_checkpoint_path)
                cur.execute(insert_row(state_table_name,task,iteration,epoch,curr_acc_test=acc_test,curr_acc_train=acc_train,curr_loss_test=loss_test,curr_loss_train=loss_train))
                conn.commit()
                print('epoch [{}/{}], loss:{:.4f}, acc {}/{} = {:.4f}%, time: {}'.format(epoch, args.epochs, loss_test, correct,total,acc_test*100, curr_exec_time))        
                if no_improve_it == args.earlystopping_it:
                    break
            except KeyboardInterrupt as e:
                print(e)
                print("GOODBY :)))")
                return False
        # if iteration == (args.iterations):
        #     writer.add_hparams({"key":data_composition_key,"ss":ss},{"hparam/accuracy":best_acc,"hparam/loss":best_loss,"hparam/execution_time":best_exec_time})
    return True


def start_task_listener(args):
    

    connection = pika.BlockingConnection(
        pika.ConnectionParameters(
            host=args.rabbitmq_server,
            heartbeat=0
        )
    )

    channel = connection.channel()

    channel.queue_declare(queue="task_queue",durable=True)

    print(" [*] Waiting for tasks. To exit press CTRL+C")

    def callback(ch,method,properties,body):
        task = body.decode("utf-8")
        print(" [x] Received " + task)
        finished_successfully = False
        try:
            conn = psycopg2.connect(host=args.database_host,database=args.database,user=args.database_user,password=args.database_password)
        except Exception as e:
            raise e
        while not finished_successfully:
            try:
                finished_successfully = analysis(conn,args,task)
                conn.close()
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('| WARNING: ran out of memory, halfing batch size')
                    args.batch_size=max(1,args.batch_size//2)
                    print(f"Batch size is now: {args.batch_size}")
                    continue
                else:
                    raise e
        
        print(" [x] Done |:->")
        ch.basic_ack(delivery_tag=method.delivery_tag)

    channel.basic_qos(prefetch_count=1)#this is important to prefetch only one task <- heavily influence the way the tasks are spread over the cluster

    channel.basic_consume(queue="task_queue",on_message_callback=callback)

    channel.start_consuming()