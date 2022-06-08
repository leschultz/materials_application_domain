import torch
import numpy as np

class Loss_queue:
    def __init__(self, capacity = 3):
        self.capacity = capacity
        self.loss_array = []
    def append(self, loss):
        """return true if full"""
        if len(self.loss_array) == 0 or loss == self.loss_array[0]:
            # print("Append:", loss)
            self.loss_array.append(loss)
        else:
            self.loss_array = [loss]

        if len(self.loss_array) == self.capacity:
            return True
        else:
            return False



def fit(loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, interval, metrics=[],
        start_epoch=0):

    loss_queue = Loss_queue()

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):


        # Train stage
        train_loss = train_epoch(loader,
                                          model,
                                          loss_fn,
                                          optimizer, cuda, interval)

        # message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)

        # print(message)

        scheduler.step()

        if loss_queue.append( round(train_loss,4)):
            # loss are the same for 4 times
            break

def fit2(loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, interval, metrics=[],
        start_epoch=0):

    loss_queue = Loss_queue()

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):


        # Train stage
        train_loss = train_epoch2(loader,
                                          model,
                                          loss_fn,
                                          optimizer, cuda, interval)

        # message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)

        # print(message)

        scheduler.step()

        if loss_queue.append( round(train_loss,4)):
            # loss are the same for 5 times
            break
def fit_transformer(loader, trainer, optimizer, n_epochs, cuda, interval, metrics=[],
        start_epoch=0):

    loss_queue = Loss_queue()

    for i in range(n_epochs):

        losses = train1epoch(loader, trainer, optimizer, interval)

        if loss_queue.append( round(losses,3)):
            break

def train1epoch(train_loader, trainer, optimizer, interval):
    trainer.transformer.float()
    losses = []
    for idx, data in enumerate(train_loader):

        loss = trainer(data) # do 1 step training
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
    return sum(losses) / len(losses)

def train_epoch(train_loader, model, loss_fn, optimizer, cuda, interval):


    model.train()
    losses = []
    total_loss = 0

    for batch_idx, data in enumerate(train_loader):
#         target = target if len(target) > 0 else None
#         if not type(data) in (tuple, list):
#             data = (data,)
#         if cuda:
#             data = tuple(d.cuda() for d in data)
#             if target is not None:
#                 target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
#         if target is not None:
#             target = (target,)
#             loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
#         print(loss_outputs )
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
#         print(type(loss))
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()


        if batch_idx % interval == 0:
            # message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     batch_idx * len(data[0]), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), np.mean(losses))
            #
            # print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss

def train_epoch2(train_loader, model, criterion, optimizer, cuda, interval):


    model.train()
    losses = []
    total_loss = 0

    for batch_idx, data in enumerate(train_loader):

        optimizer.zero_grad()
        p1, p2, z1, z2  = model(*data)

        loss_outputs = loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
#         print(loss_outputs )
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
#         print(type(loss))
        losses.append(loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if batch_idx % interval == 0:
            # message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     batch_idx * len(data[0]), len(train_loader.dataset),
            #     100. * batch_idx / len(train_loader), np.mean(losses))
            #
            # print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss
