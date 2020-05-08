import torch
from utils.consts import nr_of_classes

def test(model, test_data_loader,criterion,optimizer,batch_size):
    model.eval()
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