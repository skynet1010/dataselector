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

from code.utils.dataloader_provider import get_dataloaders
from code.utils.postgres_functions import table_row_sql, insert_row, update_row, make_sure_table_exist
from code.utils.consts import optimizer_dict, loss_dict
from code.selector import calc_metrics

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
    avg_p_c = 0
    avg_n_c = 0

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
            tp += (pred_cpu[label_ones_idx]==labels[label_ones_idx]).sum().item()
            fp += (pred_cpu[label_ones_idx]!=labels[label_ones_idx]).sum().item()
            tn += (pred_cpu[label_zeroes_idx]==labels[label_zeroes_idx]).sum().item()
            fn += (pred_cpu[label_zeroes_idx]!=labels[label_zeroes_idx]).sum().item()
            avg_p_c += confidence[predicted==1]
            avg_n_c += confidence[predicted==0]

    metrics = {"acc":correct/total, "loss":running_loss/total,"TP":tp,"FN":fn,"FP":fp,"TN":tn,"AVG_PC":avg_p_c/total,"AVG_NC":avg_n_c/total}
                        
    return model,metrics

def evaluate(
    model: torch.nn.Module,
    eval_data_loader: torch.utils.data.DataLoader,
    parameters: Dict, 
    device: torch.device,
    dtype: torch.dtype
) -> float:
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
    avg_p_c = 0
    avg_n_c = 0

    with torch.no_grad():
        for data in eval_data_loader:
            tmp_batch_size = len(data["labels"])
            lbl_onehot = torch.FloatTensor(tmp_batch_size,2).to(device=device,dtype=dtype)
            # =============datapreprocessing=================
            img = torch.FloatTensor(data["imagery"]).to(device=device,dtype=dtype)
            # ===================forward=====================
            output = model(img)
            
            lbl_onehot.zero_()
            :torch.optim.SGD
            confidence, predicted = torch.max(out_softmax.data, 1)
            total += tmp_batch_size

            labels = data["labels"].view(tmp_batch_size)
            pred_cpu = predicted.cpu()
            correct += (pred_cpu == labels).sum().item()

            label_ones_idx = labels.nonzero()
            label_zeroes_idx = (labels==0).nonzero()
            tp += (pred_cpu[label_ones_idx]==labels[label_ones_idx]).sum().item()
            fp += (pred_cpu[label_ones_idx]!=labels[label_ones_idx]).sum().item()
            tn += (pred_cpu[label_zeroes_idx]==labels[label_zeroes_idx]).sum().item()
            fn += (pred_cpu[label_zeroes_idx]!=labels[label_zeroes_idx]).sum().item()

            avg_p_c += confidence[predicted==1]
            avg_n_c += confidence[predicted==0]
    metrics = {"acc":correct/total, "loss":running_loss/total,"TP":tp,"FN":fn,"FP":fp,"TN":tn,"AVG_PC":avg_p_c/total,"AVG_NC":avg_n_c/total}
    return metrics

def hyperparameter_optimization(args:Namespace,conn:connection,task:str):
    dtype = torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cur = conn.cursor()


    _,ss,data_composition_key,model_key=task.split(":")

    train_data_loader, valid_data_loader, test_data_loader = get_dataloaders(args,ss,data_composition_key, model_key)

    make_sure_table_exist(args, conn, cur, args.states_current_task_table_name)
    make_sure_table_exist(args, conn, cur, args.best_validation_results_table_name)
    make_sure_table_exist(args, conn, cur, args.best_test_results_table_name)

    range_lr = ax.RangeParameter(name="lr",lower=1e-7,upper=0.5,parameter_type=ax.ParameterType.FLOAT)
    range_weight_decay = ax.RangeParameter(name="weight_decay",lower=1e-8,upper=0.5,parameter_type=ax.ParameterType.FLOAT)
    choice_optimizer = ax.ChoiceParameter(name="optimizer", values=["Adadelta","Adagrad","Adam","AdamW","Adamax","ASGD","RMSprop","Rprop","SGD"], parameter_type=ax.ParameterType.STRING)
    choice_criterion = ax.ChoiceParameter(name="criterion",values=["BCELoss","MSELoss"],parameter_type=ax.ParameterType.STRING)

    search_space = SearchSpace(parameters=[range_lr, range_weight_decay,choice_optimizer,choice_criterion])

    experiment = ax.Experiment(name="experiment_building_blocks",search_space=search_space)

    sobol = Models.SOBOL(search_space=experiment.search_space)
    generator_run = sobol.gen(5)
        
    return True