from argparse import Namespace
from psycopg2.extensions import connection

import torch
from torch import nn
import numpy as np
import ax

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
from typing import Dict
import time

from code.utils.dataloader_provider import get_dataloaders
from code.utils.postgres_functions import table_row_sql, insert_row, update_row, make_sure_table_exist
from code.utils.consts import optimizer_dict, loss_dict, data_compositions
from code.selector import calc_metrics
from code.utils.model_manipulator import manipulateModel

conn=None
cur = None
args = None
ss = None
data_composition_key = None
model_key = None



def train(
    model: torch.nn.Module,
    train_data_loader: torch.utils.data.DataLoader, 
    parameters: Dict,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    
    model.to(device=device,dtype=dtype)
    model.train()

    criterion = loss_dict[parameters.get("criterion","MSELoss")]
    optimizer = optimizer_dict[parameters.get("optimizer","Adam")](model.parameters(), lr=parameters.get("lr",1e-3),weight_decay=parameters.get("weight_decay",1e-5))
    
    running_loss = 0
    correct= 0
    total=0

    softmax = torch.nn.Softmax(dim=1)

    tp = 0
    fn = 0
    fp = 0
    tn = 0
    tp_c = 0
    fp_c = 0
    tn_c = 0
    fn_c = 0


    num_epochs = parameters.get("num_epochs", 20)
    print(num_epochs)
    for e in range(num_epochs):
        for step,data in enumerate(train_data_loader):
            tmp_batch_size = len(data["labels"])
            lbl_onehot = torch.FloatTensor(tmp_batch_size,2).to(device=device,dtype=dtype)

            # =============datapreprocessing=================
            img = torch.FloatTensor(data["imagery"].float()).to(device=device,dtype=dtype)
            lbl_onehot.zero_()
            lbl_onehot = lbl_onehot.scatter(1,data["labels"].to(device=device,dtype=torch.long),1).to(device=device,dtype=dtype)
            # ===================forward=====================
            output = model(img)
            loss = criterion(output, lbl_onehot)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss+=(loss.item()*tmp_batch_size)
            #determine acc
            out_softmax = torch.nn.Softmax(dim=1)

            confidence, predicted = torch.max(out_softmax, 1)
            total += tmp_batch_size
            labels = data["labels"].view(tmp_batch_size)
            pred_cpu = predicted.cpu()
            correct += (pred_cpu == labels).sum().item()

            label_ones_idx = labels.nonzero()
            label_zeroes_idx = (labels==0).nonzero()
            tp_idx = pred_cpu[label_ones_idx]==labels[label_ones_idx]
            tp += (tp_idx).sum().item()
            fp_idx = pred_cpu[label_ones_idx]!=labels[label_ones_idx]
            fp += (fp_idx).sum().item()
            tn_idx = pred_cpu[label_zeroes_idx]==labels[label_zeroes_idx]
            tn += (tn_idx).sum().item()
            fn_idx = pred_cpu[label_zeroes_idx]!=labels[label_zeroes_idx] 
            fn += (fn_idx).sum().item()
            tp_c += confidence[tp_idx].sum().item()
            fp_c += confidence[fp_idx].sum().item()
            tn_c += confidence[tn_idx].sum().item()
            fn_c += confidence[fn_idx].sum().item()

    metrics = {"acc":correct/total, "loss":running_loss/total,"TP":tp,"FN":fn,"FP":fp,"TN":tn,"TPC":tp_c/total,"FPC":fp_c/total,"TNC":tn_c/total,"FNC":fn_c/total}
                        
    return model,metrics

def evaluate(
    model: torch.nn.Module,
    eval_data_loader: torch.utils.data.DataLoader,
    parameters: Dict, 
    device: torch.device,
    dtype: torch.dtype
) -> Dict:
    model.to(device=device,dtype=dtype)

    model.eval()

    criterion = loss_dict[parameters.get("criterion","MSELoss")]


    correct = 0
    total = 0
    running_loss=0

    softmax = torch.nn.Softmax(dim=1)

    tp = 0
    fn = 0
    fp = 0
    tn = 0
    tp_c = 0
    fp_c = 0
    tn_c = 0
    fn_c = 0

    with torch.no_grad():
        for data in eval_data_loader:
            tmp_batch_size = len(data["labels"])
            lbl_onehot = torch.FloatTensor(tmp_batch_size,2).to(device=device,dtype=dtype)
            # =============datapreprocessing=================
            img = torch.FloatTensor(data["imagery"]).to(device=device,dtype=dtype)
            # ===================forward=====================
            
            output = model(img)
            
            out_softmax = softmax(output)

            lbl_onehot.zero_()
            loss = criterion(output, lbl_onehot)

            running_loss+=(loss.item()*tmp_batch_size)
            confidence, predicted = torch.max(out_softmax.data, 1)
            total += tmp_batch_size

            labels = data["labels"].view(tmp_batch_size)
            pred_cpu = predicted.cpu()
            correct += (pred_cpu == labels).sum().item()

            label_ones_idx = labels.nonzero()
            label_zeroes_idx = (labels==0).nonzero()
            tp_idx = pred_cpu[label_ones_idx]==labels[label_ones_idx]
            tp += (tp_idx).sum().item()
            fp_idx = pred_cpu[label_ones_idx]!=labels[label_ones_idx]
            fp += (fp_idx).sum().item()
            tn_idx = pred_cpu[label_zeroes_idx]==labels[label_zeroes_idx]
            tn += (tn_idx).sum().item()
            fn_idx = pred_cpu[label_zeroes_idx]!=labels[label_zeroes_idx] 
            fn += (fn_idx).sum().item()
            tp_c += confidence[tp_idx].sum().item()
            fp_c += confidence[fp_idx].sum().item()
            tn_c += confidence[tn_idx].sum().item()
            fn_c += confidence[fn_idx].sum().item()

    metrics = {"acc":correct/total, "loss":running_loss/total,"TP":tp,"FN":fn,"FP":fp,"TN":tn,"TPC":tp_c/total,"FPC":fp_c/total,"TNC":tn_c/total,"FNC":fn_c/total}
    return metrics

def objective(parameters):
    train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(args,ss,data_composition_key, model_key)
    model = manipulateModel(model_key,args.is_feature_extraction,data_compositions[data_composition_key])
    criterion = parameters.get("criterion")
    optimizer = parameters.get("optimizer")(model.parameters(), lr=parameters.get("lr"),weight_decay=parameters.get("weight_decay"))
    for epoch in range(args.epochs+1):
        start = time.time()
        model,train_metrics = train(model,train_data_loader,criterion,optimizer,args.batch_size) 
        valid_metrics =  evaluate(model,valid_data_loader,criterion,optimizer,args.batch_size)

        train_metrics = calc_metrics(train_metrics)
        valid_metrics = calc_metrics(valid_metrics)

        curr_exec_time = time.time()-start
        train_metrics["exec_time"] = curr_exec_time
        if valid_metrics["acc"] > best_acc:
            best_acc = valid_metrics["acc"]
            best_loss = valid_metrics["loss"]
            update=True
        elif valid_metrics["acc"] == best_acc and best_loss < valid_metrics["loss"]:
            best_loss = valid_metrics["loss"]
            update=True
        elif valid_metrics["acc"] == best_acc and best_loss == valid_metrics["loss"] and curr_exec_time<best_exec_time:
            update=True
        if update:
            best_acc_curr_iteration = valid_metrics["acc"]
            best_loss_curr_iteration = valid_metrics["loss"]
            no_improve_it = 0
            best_exec_time = curr_exec_time
            valid_metrics["exec_time"]=best_exec_time
            torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict()}, best_checkpoint_path)
            cur.execute(update_row(args.best_validation_results_table_name,task,iteration,epoch,valid_metrics))
            conn.commit()
            update=False
        elif valid_metrics["acc"] > best_acc_curr_iteration or valid_metrics["loss"] < best_loss_curr_iteration:
            best_acc_curr_iteration = valid_metrics["acc"]
            best_loss_curr_iteration = valid_metrics["loss"]
            no_improve_it = 0
        else:
            no_improve_it+=1
        torch.save({"epoch":epoch,"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict()}, state_checkpoint_path)
        cur.execute(insert_row(args.states_current_task_table_name,args, task,iteration,epoch,timestamp=time.time(),m1=valid_metrics,m2=train_metrics))
        conn.commit()
        print('epoch [{}/{}], loss:{:.4f}, {:.4f}%, time: {}'.format(epoch, args.epochs, valid_metrics["loss"],valid_metrics["acc"]*100, curr_exec_time))        
        if no_improve_it == args.earlystopping_it:
            break
    # except Exception as e:
    #     print(f"Exception occured in iteration {iteration}, epoch {epoch}",e)
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
    test_metrics =  test(model,test_data_loader,criterion,optimizer,args.batch_size)

    test_metrics = calc_metrics(test_metrics)
    test_metrics["exec_time"] = time.time()-start
    
    cur.execute(insert_row(args.best_test_results_table_name, args, task, iteration, -1, timestamp=time.time(),m1=test_metrics))

    conn.commit()
    except KeyboardInterrupt as e:
        print(e)
        print("GOODBY :)))")
        return False

    return True

def hyperparameter_optimization(a:Namespace,c:connection,task:str):
    dtype = torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global cur
    cur = c.cursor()
    global conn
    conn = c
    global args
    args = a

    global ss
    global data_composition_key
    global model_key
    _,ss,data_composition_key,model_key=task.split(":")

    make_sure_table_exist(args, conn, cur, args.states_current_task_table_name)
    make_sure_table_exist(args, conn, cur, args.best_validation_results_table_name)
    make_sure_table_exist(args, conn, cur, args.best_test_results_table_name)

    range_lr = ax.RangeParameter(name="lr",lower=1e-7,upper=0.5,parameter_type=ax.ParameterType.FLOAT)
    range_weight_decay = ax.RangeParameter(name="weight_decay",lower=1e-8,upper=0.5,parameter_type=ax.ParameterType.FLOAT)
    choice_optimizer = ax.ChoiceParameter(name="optimizer", values=["Adadelta","Adagrad","Adam","AdamW","Adamax","ASGD","RMSprop","Rprop","SGD"], parameter_type=ax.ParameterType.STRING)
    choice_criterion = ax.ChoiceParameter(name="criterion",values=["BCELoss","MSELoss"],parameter_type=ax.ParameterType.STRING)

    search_space = ax.SearchSpace(parameters=[range_lr, range_weight_decay,choice_optimizer,choice_criterion])

    experiment = ax.Experiment(name="experiment_building_blocks",search_space=search_space)

    sobol = ax.Models.SOBOL(search_space=experiment.search_space)
    generator_run = sobol.gen(1)
        
    return True