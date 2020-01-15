"""
    The calculation to be performed at each point (modified model), evaluating
    the loss value, accuracy and eigen values of the hessian matrix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd.variable import Variable
import pdb



def loss_fn_kd(student_logits, teacher_logits, T):
    """
    Compute the knowledge-distillation (KD) loss given outputs from student and teacher
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """


    teacher_soft_logits = F.softmax(teacher_logits / T, dim=1)

    teacher_soft_logits = teacher_soft_logits.float()
    student_soft_logits = F.log_softmax(student_logits/T, dim=1)


    #For KL(p||q), p is the teacher distribution (the target distribution), and
    KD_loss = nn.KLDivLoss(reduction='batchmean')(student_soft_logits, teacher_soft_logits)
    KD_loss = (T ** 2) * KD_loss
    # KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T)

    return KD_loss
def eval_loss(net, criterion, loader, use_cuda=False):
    """
    Evaluate the loss value for a given 'net' on the dataset provided by the loader.

    Args:
        net: the neural net model
        criterion: loss function
        loader: dataloader
        use_cuda: use cuda or not
    Returns:
        loss value and accuracy
    """
    correct = 0
    total_loss = 0
    total = 0 # number of samples
    num_batch = len(loader)

    if use_cuda:
        net.cuda()
    net.eval()
    pdb.set_trace()
    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                # inputs = Variable(inputs)
                # targets = Variable(targets)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()

        elif isinstance(criterion, nn.MSELoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)

                one_hot_targets = torch.FloatTensor(batch_size, 10).zero_()
                one_hot_targets = one_hot_targets.scatter_(1, targets.view(batch_size, 1), 1.0)
                one_hot_targets = one_hot_targets.float()
                one_hot_targets = Variable(one_hot_targets)
                if use_cuda:
                    inputs, one_hot_targets = inputs.cuda(), one_hot_targets.cuda()
                outputs = F.softmax(net(inputs))
                loss = criterion(outputs, one_hot_targets)
                total_loss += loss.item()*batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.cpu().eq(targets).sum().item()

    return total_loss/total, 100.*correct/total

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res[0]

def test_model(model, criterion, data_loader, printing=False, eta=None, teacher_model=None, topk=1):

    #switch to eval mode
    # if printing:
    #     print('Evaluating Model...')
    # pdb.set_trace()
    model.cuda()
    model.eval()


    n_test = 0.
    n_correct = 0.
    loss = 0.
    kl_loss = 0.
    for iter, (inputs, target) in enumerate(data_loader):
        n_batch = inputs.size()[0]
        # print(iter)
        inputs = inputs.cuda()
        target = target.cuda()

        if eta is not None:
            output = model(inputs, eta=eta)
        else:
            output = model(inputs)

        loss += criterion(output, target).item()
        # acc_ = accuracy(output, target)

        n_correct += accuracy(output, target, topk=(topk,)).item() * n_batch/100.
        n_test += n_batch

        if teacher_model is not None:
            with torch.no_grad():
                teacher_output = teacher_model(inputs)
                kl_loss += loss_fn_kd(output, teacher_output, T=1)
                # print(kl_loss)
            # print(kl_loss.size())
            # exit()

    test_accuracy= 100*n_correct/n_test
    test_loss = 128*loss/n_test
    kl_loss = 128*kl_loss/n_test



    if printing:
        print('Test Accuracy %.3f'% test_accuracy)
        print('Test Loss %.3f'% test_loss)

    #Revert model to training mode before exiting
    model.train()

    if teacher_model is None:
        return test_loss, test_accuracy
    else:
        return test_loss, test_accuracy, kl_loss

