import torch
from utils.consts import data_compositions, model_dict, learning_rate, nr_of_classes, time_stamp
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

def get_retrain_model_param(args,cur, model,optimizer,state_checkpoint_path,task,retrain):
    best_acc_curr_iteration = 0.0
    best_loss_curr_iteration = sys.float_info.max
    if retrain:
        if os.path.isfile(state_checkpoint_path):
            state_checkpoint = torch.load(state_checkpoint_path)
            model.load_state_dict(state_checkpoint["model_state_dict"])
            optimizer.load_state_dict(state_checkpoint["optimizer_state_dict"])
            print("Model loaded from file.")
        cur.execute(table_row_sql(args.states_current_task_table_name, args, task))
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
    try:
        cur = conn.cursor()

        ss,data_composition_key,model_key=task.split(":")

        train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(args,ss,data_composition_key, model_key)


        make_sure_table_exist(args, conn, cur, args.states_current_task_table_name)
        make_sure_table_exist(args, conn, cur, args.best_validation_results_table_name)
        make_sure_table_exist(args, conn, cur, args.best_test_results_table_name)
 
        retrain, niteration, nepoch, best_acc, best_loss, best_exec_time = init_analysis_params(args,conn,cur,task)

        res_path = get_valid_path(args,data_composition_key,ss)

        best_checkpoint_path = os.path.join(res_path,f"best_{model_key}.pth")

        state_checkpoint_path = os.path.join(res_path,f"state_{model_key}.pth")

        for iteration in range(niteration, args.iterations+1):
            print(task,iteration)
            
            model = manipulateModel(model_key,args.is_feature_extraction,data_compositions[data_composition_key])
            #return True
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

            model,optimizer,best_acc_curr_iteration,best_loss_curr_iteration,retrain = get_retrain_model_param(args,cur,model,optimizer,state_checkpoint_path,task,retrain)

            update = False
            no_improve_it = 0
            for epoch in range(nepoch,args.epochs+1):
                start = time.time()
                loss_train,acc_train = train(model,train_data_loader,criterion,optimizer,args.batch_size) 
                loss_valid,correct, total =  test(model,valid_data_loader,criterion,optimizer,args.batch_size)
                curr_exec_time = time.time()-start
                acc_valid = correct/total

                if acc_valid > best_acc:
                    best_acc = acc_valid
                    best_loss = loss_valid
                    update=True
                elif acc_valid == best_acc and best_loss < loss_valid:
                    best_loss = loss_valid
                    update=True
                elif acc_valid == best_acc and best_loss == loss_valid and curr_exec_time<best_exec_time:
                    update=True
                if update:
                    best_acc_curr_iteration = acc_valid
                    best_loss_curr_iteration = loss_valid
                    no_improve_it = 0
                    best_exec_time = curr_exec_time
                    torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict()}, best_checkpoint_path)
                    cur.execute(update_row(args.best_validation_results_table_name,task,iteration,epoch,best_acc,best_loss,best_exec_time))
                    conn.commit()
                    update=False
                elif acc_valid > best_acc_curr_iteration or loss_valid < best_loss_curr_iteration:
                    best_acc_curr_iteration = acc_valid
                    best_loss_curr_iteration = loss_valid
                    no_improve_it = 0
                else:
                    no_improve_it+=1
                torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict()}, state_checkpoint_path)
                cur.execute(insert_row(args.states_current_task_table_name,args, task,iteration,epoch,curr_acc_valid=acc_valid,curr_acc_train=acc_train,curr_loss_valid=loss_valid,curr_loss_train=loss_train,timestamp=time.time()))
                conn.commit()
                print('epoch [{}/{}], loss:{:.4f}, acc {}/{} = {:.4f}%, time: {}'.format(epoch, args.epochs, loss_valid, correct,total,acc_valid*100, curr_exec_time))        
                if no_improve_it == args.earlystopping_it:
                    break
            #TODO load best model ;)
            try:
                os.remove(state_checkpoint_path)
            except Exception as e:
                print("Deleting state dict failed",e)

            model = manipulateModel(model_key,args.is_feature_extraction,data_compositions[data_composition_key])
            if not os.path.isfile(best_checkpoint_path):
                print("Best checkpoint file does not exist!!!")
                return True
            
            best_checkpoint = torch.load(best_checkpoint_path)
            model.load_state_dict(best_checkpoint["model_state_dict"])
            optimizer.load_state_dict(best_checkpoint["optimizer_state_dict"])
            start = time.time()
            loss_test,correct, total =  test(model,test_data_loader,criterion,optimizer,args.batch_size)
            
            cur.execute(insert_row(args.best_test_results_table_name, args, task, iteration, -1, correct/total, loss_test, time.time()-start),timestamp=time.time())

            conn.commit()

    except KeyboardInterrupt as e:
        print(e)
        print("GOODBY :)))")
        return False

    return True
