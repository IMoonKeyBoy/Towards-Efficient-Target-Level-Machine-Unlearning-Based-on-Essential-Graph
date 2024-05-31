import sys
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%%

class base_warmup():
    def __init__(self, optimizer, warm_step, warm_lr, dst_lr):
        ''' base class for warmup scheduler
        Args:
            optimizer: adjusted optimizer
            warm_step: total number of warm_step,(batch num)
            warm_lr: start learning rate of warmup
            dst_lr: init learning rate of train stage eg. 0.01
        '''
        assert warm_lr < dst_lr, "warmup lr must smaller than init lr"
        self.optimizer = optimizer
        self.warm_lr = warm_lr
        self.init_lr = dst_lr
        self.warm_step = warm_step
        self.stepped = 0
        if self.warm_step:
            self.optimizer.param_groups[0]['lr'] = self.warm_lr

    def step(self):
        self.stepped += 1

    def if_in_warmup(self) -> bool:
        return True if self.stepped < self.warm_step else False

    def set2initial(self):
        ''' Reset the learning rate to initial lr of training stage '''
        self.optimizer.param_groups[0]['lr'] = self.init_lr

    @property
    def now_lr(self):
        return self.optimizer.param_groups[0]['lr']

class linear_warmup_scheduler(base_warmup):
    def __init__(self, optimizer, warm_step, warm_lr, dst_lr):
        super().__init__(optimizer, warm_step, warm_lr, dst_lr)
        if (self.warm_step <= 0):
            self.inc = 0
        else:
            self.inc = (self.init_lr - self.warm_lr) / self.warm_step

    def step(self) -> bool:
        if (not self.stepped < self.warm_step): return False
        self.optimizer.param_groups[0]['lr'] += self.inc
        super().step()
        return True

    def still_in_warmup(self) -> bool:
        return True if self.stepped < self.warm_step else False

class LabelSmoothCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing=smoothing
        print('label smoothing:', self.smoothing)
        
    def forward(self, pred, label):
        pred = F.softmax(pred, dim=1)
        one_hot_label = F.one_hot(label, pred.size(1)).float()
        smoothed_one_hot_label = (
            1.0 - self.smoothing) * one_hot_label + self.smoothing / pred.size(1)
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        loss = loss.sum(axis=1, keepdim=False)
        loss = loss.mean()

        return loss

def direct_unlearning(net, epochs, lr,
                      rest_trainloader, rest_testloader, unlearn_testloader, save_info='./',
                      save_acc=80.0,
                      seed=0,
                      start_epoch=0, device='cuda',
                      label_smoothing=0, warmup_step=0, warm_lr=10e-5, pth_name=""):

    rest_test_loss, rest_test_acc = test(net, rest_testloader)
    unlearning_test_loss, unlearning_test_acc = test(net, unlearn_testloader)
    print(rest_test_acc, "\t", unlearning_test_acc)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.95, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    warmup_scheduler = linear_warmup_scheduler(optimizer, warmup_step, warm_lr, lr)

    st = time.time()
    # simulate that only little data in server

    for epoch in range(start_epoch, epochs):

        net.train()

        for batch_idx, (inputs, targets) in enumerate(rest_trainloader):

            if_warmup = False if warmup_scheduler == None else warmup_scheduler.if_in_warmup()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if if_warmup:
                warmup_scheduler.step()

        if not warmup_scheduler or not warmup_scheduler.if_in_warmup():
            scheduler.step()

        rest_test_loss, rest_test_acc = test(net, rest_testloader)
        unlearning_test_loss, unlearning_test_acc = test(net, unlearn_testloader)
        print(rest_test_acc, "\t", unlearning_test_acc)

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    save_path = save_info / ("ctraining" + '.pth')
    print('save path', save_path)
    torch.save(net.state_dict(), save_path)


def retrain(net, epochs, lr,
            train_loader,
            test_loader,
            unlearn_testloader,
            save_info='./',
            seed=0,
            start_epoch=0,
            device='cuda',
            label_smoothing=0,
            warmup_step=0,
            warm_lr=10e-5):

    test_loss, test_acc = test(net, test_loader)
    unlearning_loss, unlearning_acc = test(net, unlearn_testloader)
    print(test_acc, "\t", unlearning_acc)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    warmup_scheduler = linear_warmup_scheduler(optimizer, warmup_step, warm_lr, lr)

    for epoch in range(start_epoch, epochs):
        """
        Start the training code.
        """
        net.train()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if_warmup = False if warmup_scheduler == None else warmup_scheduler.if_in_warmup()

            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            if if_warmup:
                warmup_scheduler.step()

        test_loss, test_acc = test(net, test_loader)
        unlearning_loss, unlearning_acc = test(net, unlearn_testloader)
        print(test_acc, "\t", unlearning_acc)


        if not warmup_scheduler or not warmup_scheduler.if_in_warmup():
            scheduler.step()


def train(net, epochs, lr, train_loader, test_loader, device='cuda',label_smoothing=0, warmup_step=0, warm_lr=10e-5):
    """
    Training a network
    :param net: Network for training
    :param epochs: Number of epochs in total.
    :param batch_size: Batch size for training.
    """

    st = time.time()

    #print('==> Preparing data..')
    criterion = LabelSmoothCELoss(label_smoothing)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(epochs*0.5), int(epochs*0.75)], gamma=0.1)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0)
    warmup_scheduler = linear_warmup_scheduler(optimizer, warmup_step, warm_lr, lr)

    #best_acc = 0  # best test accuracy
    for epoch in range(0, epochs):
        """
        Start the training code.
        """
        #print('\nEpoch: %d' % epoch, '/ %d;' % epochs, 'learning_rate:', optimizer.state_dict()['param_groups'][0]['lr'])
        net.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):

            # if warmup scheduler==None or not in scope of warmup -> if_warmup=False
            if_warmup=False if warmup_scheduler==None else warmup_scheduler.if_in_warmup()
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            if if_warmup:
                warmup_scheduler.step()
        if not warmup_scheduler or not warmup_scheduler.if_in_warmup():
            scheduler.step()
        print(test(net,test_loader))

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

def test(net, testloader):
    criterion = nn.CrossEntropyLoss()

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    num_val_steps = len(testloader)
    val_acc = correct / total
    # print(val_acc)
    loss = test_loss / num_val_steps
    return loss, val_acc

def load_model_pytorch(model, load_model, model_name):
    # print("=> loading checkpoint '{}'".format(load_model))
    checkpoint = torch.load(load_model)

    if 'state_dict' in checkpoint.keys():
        load_from = checkpoint['state_dict']
    else:
        load_from = checkpoint

    # match_dictionaries, useful if loading model without gate:
    if 'module.' in list(model.state_dict().keys())[0]:
        if 'module.' not in list(load_from.keys())[0]:
            from collections import OrderedDict

            load_from = OrderedDict([("module.{}".format(k), v) for k, v in load_from.items()])

    if 'module.' not in list(model.state_dict().keys())[0]:
        if 'module.' in list(load_from.keys())[0]:
            from collections import OrderedDict

            load_from = OrderedDict([(k.replace("module.", ""), v) for k, v in load_from.items()])

    # just for vgg
    if model_name == "vgg":
        from collections import OrderedDict

        load_from = OrderedDict([(k.replace("features.", "features"), v) for k, v in load_from.items()])
        load_from = OrderedDict([(k.replace("classifier.", "classifier"), v) for k, v in load_from.items()])

    if 1:
        for ind, (key, item) in enumerate(model.state_dict().items()):
            if ind > 10:
                continue
            # print(key, model.state_dict()[key].shape)
        # print("*********")

        for ind, (key, item) in enumerate(load_from.items()):
            if ind > 10:
                continue
            # print(key, load_from[key].shape)

    for key, item in model.state_dict().items():
        # if we add gate that is not in the saved file
        if key not in load_from:
            load_from[key] = item
        # if load pretrined model
        if load_from[key].shape != item.shape:
            load_from[key] = item

    model.load_state_dict(load_from, False)

