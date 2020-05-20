import torch
from code.utils.consts import nr_of_classes

def train(model, train_data_loader, criterion, optimizer,batch_size):
    model.train()
    running_loss = 0
    correct= 0
    total=0
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    avg_p_c = 0
    avg_n_c = 0
    softmax = torch.nn.Softmax(dim=1)

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
        running_loss+=(loss.item()*tmp_batch_size)
        #determine acc
        out_softmax = softmax(output)

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

    return metrics