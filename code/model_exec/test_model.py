import torch
from code.utils.consts import nr_of_classes


def test(model, test_data_loader,criterion,optimizer,batch_size):
    model.eval()
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

            out_softmax = softmax(output)

            running_loss+=(loss.item()*tmp_batch_size)
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