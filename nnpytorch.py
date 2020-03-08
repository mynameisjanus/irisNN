import numpy as np
import torch
# import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

# Specify the model

criterion = nn.NLLLoss()

def train_model(data, model, lr = 0.01, momentum = 0.9, nesterov = False, n_epochs = 30):
        """
        Train a model for N epochs.
        """
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum, nesterov)
        
        for epoch in range(n_epochs):
                        
                # print("----------\nEpoch {}:\n".format(epoch))
                
                loss, acc = run_epoch(data, model.train(), optimizer)
                # print('Train loss: {:.6f} | Train accuracy: {:.6f}'.format(loss, acc))

def run_epoch(data, model, optimizer):
        
        is_training = model.training
        
        x, y = data['x'], data['y']
        output = model(x)
        
        # Predict and calculate accuracy
        predictions = torch.argmax(output, dim = 1)
        accuracy = np.mean(np.equal(predictions.numpy(), y.numpy()))
        
        # Compute loss
        loss = F.cross_entropy(output, y)
        
        if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return loss.data.item(), accuracy