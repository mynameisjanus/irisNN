import torch.nn as nn

# Specify the model

model = nn.Sequential(
        nn.Linear(X_train.shape[1], hidden_layer),
        nn.LeakyReLU(negative_slope = 0.01),
        nn.Linear(hidden_layer, 3)
        )

def train_model(train_data, dev_data, model, lr, momentum, nesterov = False, n_epochs):
        """
        Train a model for N epochs.
        """
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum, nesterov)
        
        for epoch in range(1,11):
                print("----------\nEpoch {}:\n".format(epoch))
                
                loss, acc = run_epoch(train_data, model.train(), optimizer)
                print('Train loss: {:.6f} | Train accuracy: {:.6f}'.format(loss, acc))
                
                val_loss, val_acc = run_epoch(dev_data, model.eval(), optimizer)
                print('Validation loss: {:.6f} | Validation accuracy: {:.6f}'.format(val_loss, val_acc))
        return val_acc

def run_epoch(data, model, optimizer):
        losses = []
        accuracy = []
        
        is_training = model.training
        
        if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return avg_loss, avg_accuracy