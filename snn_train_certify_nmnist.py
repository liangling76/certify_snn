from __future__ import print_function
from torch.utils.data import dataset
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
from snn_model import *
from GetData import *
from snn_fire import set_time_step_certify
import sys

model = sys.argv[1]
time_step_certify = sys.argv[2]
start_eps = sys.argv[3]
max_eps = sys.argv[4]

if_train = True

batch_size = 128
num_epochs = 250 
finetune_epochs = 50

checkpoint_org = './ckpt/nmnist_' + model + '.pth.tar'

checkpoint_new = './ckpt/' + model + '_T_' + time_step_certify + '_eps_' + max_eps +  '.pth.tar'
checkpoint_ibp = './ckpt/' + model + '_T_' + time_step_certify + '_eps_' + max_eps +  '_ibp.pth.tar'
checkpoint_crown = './ckpt/' + model + '_T_' + time_step_certify + '_eps_' + max_eps +  '_crown.pth.tar'
checkpoint_analyze = './ckpt/analyze_' + model + '_T_' + time_step_certify + '_eps_' + max_eps +  '.pth.tar'


train_dataset = GetData('/home/dic_of_train_data', '/home/dic_of_train_label')
test_dataset = GetData('/home/dic_of_test_data', '/home/dic_of_test_data')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)



eps = float(start_eps)
max_eps = float(max_eps)
soft_eps = (max_eps - eps) / (num_epochs * 3)
mid = int(len(train_dataset) / (batch_size * 2))
gap = int(len(train_dataset) / (batch_size * 3))
soft_lst= [mid - gap, mid, mid + gap]


time_step_certify = int(time_step_certify)

if model == 'FC':
    snn = NMNIST_FC().cuda()
elif model == 'CONV':
    snn = NMNIST_CONV().cuda()

snn.load_state_dict(torch.load(checkpoint_org))

optimizer = torch.optim.SGD(snn.parameters(), lr=5e-4, momentum=0.9) 
criterion = nn.CrossEntropyLoss().cuda() 

sa = torch.LongTensor(10, 9).cuda()
for i in range(sa.shape[0]):
    for j in range(sa.shape[1]):
        if j < i:
            sa[i][j] = j
        else:
            sa[i][j] = j + 1


train_org_loss_lst = []
train_crown_loss_lst = []
train_robust_error_lst = []
train_diff_per_step_lst = []

test_acc_lst = []
test_loss_lst = []
test_robust_error_crown_lst = []
test_robust_error_ibp_lst = []
test_loss_crown_lst = []
test_loss_ibp_lst = []

best_acc = 0  
best_ibp = 1
best_crown = 1


torch.autograd.set_detect_anomaly(True)
for epoch in range(num_epochs + finetune_epochs + 1):

    if if_train and epoch > 0:
        snn.train()

        running_loss1 = 0
        running_loss2 = 0
        robust_error = 0
        diff_per_step = 0

        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):

            if isinstance(eps, float) and i in soft_lst and epoch <= num_epochs:
                eps += soft_eps

            snn.zero_grad()
            optimizer.zero_grad()
            
            labels = labels.cuda()
            images = images.float().cuda()

            outputs = snn(images)
            loss1 = criterion(outputs, labels)
            
            set_time_step_certify(time_step_certify)
            snn.get_para()
            crown_l, ibp_l = snn.snn_crown_ibp(images, eps, labels)
            bound_l = crown_l 
            robust_error += torch.sum((bound_l<=0).any(dim=1)).detach()

            # certify train
            lb_s = torch.zeros(bound_l.shape[0], 10).cuda()
            bound_l = lb_s.scatter(1, sa[labels], bound_l)
            loss2 = criterion(-bound_l, labels)

            loss2.backward()
            torch.nn.utils.clip_grad_norm_(snn.parameters(), 0.25) 

            optimizer.step()

            running_loss1 += loss1.detach()
            running_loss2 += loss2.detach()
            diff_per_step += snn.diff.detach()

            if (i+1) % 300 == 0:
                total_img = 300 * batch_size
                print('Epoch [%d/%d], Step [%d/%d], Loss1: %.5f, Loss2: %.5f, robust_error: %.5f, diff_per_step: %.5f, eps: %.5f' % (
                epoch, num_epochs+finetune_epochs, i + 1, len(train_dataset) // batch_size, 
                running_loss1 / 300, running_loss2 / 300, robust_error / total_img, diff_per_step / (time_step_certify * total_img), eps))
                
                train_org_loss_lst.append((running_loss1 / 300).cpu().data)
                train_crown_loss_lst.append((running_loss2 / 300).cpu().data)
                train_robust_error_lst.append((robust_error / total_img).cpu().data)
                train_diff_per_step_lst.append((diff_per_step / (time_step_certify * total_img)).cpu().data)

                running_loss1 = 0
                running_loss2 = 0
                robust_error = 0
                diff_per_step = 0

                print('Time elasped:', time.time()-start_time)
                

    snn.eval()
    
    loss = 0
    loss_crown = 0
    loss_ibp = 0
    correct = 0
    robust_error_crown = 0
    robust_error_ibp = 0
    total = 0
    test_diff = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # print(batch_idx)
            images = images.cuda()
            labels = labels.cuda()

            outputs = snn(images)
            _, predicted = outputs.cpu().max(1)

            loss += criterion(outputs, labels)
            total += float(labels.size(0))
            correct += float(predicted.eq(labels.cpu()).sum().item())

            set_time_step_certify(10)
            snn.get_para()
            crown_l, ibp_l = snn.snn_crown_ibp(images, max_eps, labels)
            robust_error_crown += torch.sum((crown_l<=0).any(dim=1))
            robust_error_ibp += torch.sum((ibp_l<=0).any(dim=1))

            lb_s = torch.zeros(images.shape[0], 10).cuda()
            bound_l = lb_s.scatter(1, sa[labels], crown_l)
            loss_crown += criterion(-bound_l, labels)

            lb_s = torch.zeros(images.shape[0], 10).cuda()
            bound_l = lb_s.scatter(1, sa[labels], ibp_l)
            loss_ibp += criterion(-bound_l, labels)

            test_diff += snn.diff.detach()



    acc = 100. * float(correct) / total
    loss /= (total / batch_size)
    loss_crown /= (total / batch_size)
    loss_ibp /= (total / batch_size)
    robust_error_crown = robust_error_crown.float() / total
    robust_error_ibp = robust_error_ibp.float() / total

    print('Iters:', epoch)
    print('loss:     %.5f' % loss.cpu().item())
    print('test acc: %.3f' % acc)
    print('loss_crown:         %.5f' % loss_crown.item())
    print('robust error crown: %.5f' % robust_error_crown.item())
    print('loss_ibp:           %.5f' % loss_ibp.item())
    print('robust error ibp:   %.5f' % robust_error_ibp.item())
    print('test diff:          %.5f' % (test_diff.cpu().item() * 1.0 / (total * 10)))
    print('\n')

    test_acc_lst.append(acc)
    test_loss_lst.append(loss.cpu().data)
    test_robust_error_crown_lst.append(robust_error_crown.cpu().data)
    test_loss_crown_lst.append(loss_crown.cpu().data)
    test_robust_error_ibp_lst.append(robust_error_ibp.cpu().data)
    test_loss_ibp_lst.append(loss_ibp.cpu().data)

    if if_train:

        torch.save(snn.state_dict(), checkpoint_new)

        analyze_dict = {
            "train_org_loss": train_org_loss_lst,
            "train_crown_loss": train_crown_loss_lst,
            "train_robust_error": train_robust_error_lst,
            "test_acc": test_acc_lst,
            "test_org_loss": test_loss_lst,
            "test_crown_loss": test_loss_crown_lst,
            "test_crown_robust_error": test_robust_error_crown_lst,
            "test_ibp_loss": test_loss_ibp_lst,
            "test_ibp_robust_error": test_robust_error_ibp_lst
        }
        torch.save(analyze_dict, checkpoint_analyze)
        print(checkpoint_analyze)

        if robust_error_crown < best_crown:
            torch.save(snn.state_dict(), checkpoint_crown)
        
        if robust_error_ibp < best_ibp:
            torch.save(snn.state_dict(), checkpoint_ibp)