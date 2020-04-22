import torch
from utils.consts import data_compositions, model_dict, learning_rate, nr_of_classes, time_stamp, state_table_name, ds_results_table_name
from utils.postgres_functions import table_row_sql, insert_row, update_row, make_sure_table_exist
from utils.model_manipulator import manipulateModel
from model_exec.train_model import train
from model_exec.test_model import test
from utils.dataloader_provider import get_dataloaders, get_results_dir
from utils.param_initializer import init_analysis_params
import os
import time
import sys
import shutil
from torch import nn

def get_retrain_model_param(args,cur, model,optimizer,state_checkpoint_path,task):
    best_acc_curr_iteration = 0.0
    best_loss_curr_iteration = sys.float_info.max
    if retrain:
        if os.path.isfile(state_checkpoint_path):
            state_checkpoint = torch.load(state_checkpoint_path)
            model.load_state_dict(state_checkpoint["model_state_dict"])
            optimizer.load_state_dict(state_checkpoint["optimizer_state_dict"])
            print("Model loaded from file.")
        cur.execute(table_row_sql(state_table_name, task))
        res = cur.fetchall()
        if res != []:
            _, _, best_acc_curr_iteration, best_loss_curr_iteration = res[0]

        retrain=False
    return model,optimizer,best_acc_curr_iteration,best_loss_curr_iteration,retrain

def get_valid_path(args,data_composition_key,ss):
    res_path = os.path.join(get_results_dir(args),f"{data_composition_key}",f"{ss}")
    if not os.path.isdir(res_path):
        try:
            os.makedirs(res_path)
        except Exception as e:
            print(e)
            exit(1)
    return res_path

def analysis(conn,args,task):
    
    cur = conn.cursor()

    ss,data_composition_key,model_key=task.split(":")

    train_data_loader, test_data_loader = get_dataloaders(args,ss,data_composition_key)


    make_sure_table_exist(args, conn, cur, state_table_name)
    make_sure_table_exist(args, conn, cur, ds_results_table_name)

    retrain, niteration, nepoch, best_acc, best_loss, best_exec_time = init_analysis_params(args,conn,cur,task)

    for iteration in range(niteration, args.iterations+1):
        
        res_path = get_valid_path(args,data_composition_key,ss)

        best_checkpoint_path = os.path.join(res_path,"best_alexnet.pth")

        state_checkpoint_path = os.path.join(res_path,"state_alexnet.pth")

        print(data_composition_key,iteration)
        
        model = manipulateModel(model_dict[model_key],args.is_feature_extraction,data_compositions[data_composition_key])

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

        model,optimizer,best_acc_curr_iteration,best_loss_curr_iteration,retrain = get_retrain_model_param(args,cur,model,optimizer,state_checkpoint_path,task)

        update = False
        no_improve_it = 0
        
        for epoch in range(nepoch,args.epochs+1):
            try:
                start = time.time()
                loss_train,acc_train = train(model,train_data_loader,criterion,optimizer,args.batch_size) 
                loss_test,correct, total =  test(model,test_data_loader,criterion,optimizer,args.batch_size)
                curr_exec_time = time.time()-start
                acc_test = correct/total

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
                cur.execute(insert_row(state_table_name, task,iteration,epoch,curr_acc_test=acc_test,curr_acc_train=acc_train,curr_loss_test=loss_test,curr_loss_train=loss_train,timestamp=time.time()))
                conn.commit()
                print('epoch [{}/{}], loss:{:.4f}, acc {}/{} = {:.4f}%, time: {}'.format(epoch, args.epochs, loss_test, correct,total,acc_test*100, curr_exec_time))        
                if no_improve_it == args.earlystopping_it:
                    break
            except KeyboardInterrupt as e:
                print(e)
                print("GOODBY :)))")
                return False
            except Exception as e:
                print(e)

    return True