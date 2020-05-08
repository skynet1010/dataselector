import torch
from utils.consts import nr_of_classes

def train(model, train_data_loader, criterion, optimizer,batch_size):
    model.train()
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
        running_loss+=(loss.item()*tmp_batch_size)
        #determine acc
        _, predicted = torch.max(output.data, 1)
        total += tmp_batch_size
        correct += (predicted.cpu() == data["labels"].view(tmp_batch_size)).sum().item()
    return running_loss/total, correct/total