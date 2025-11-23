import pickle
from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size = 296
    learning_rate = 0.01
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    loss_function = nn.BCELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    




    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)

    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch_i in range(no_epochs):
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            inputs, labels = sample['input'], sample['label']
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
        model.eval()
        epoch_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        losses.append(epoch_loss)
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            torch.save(model.state_dict(), 'saved/saved_model.pkl')
            #pickle.dump(model.state_dict(), open('saved/saved_model.pkl', 'wb')) 
            print(f'Epoch {epoch_i+1}/{no_epochs}, Loss: {epoch_loss:.4f} (improved)')       
        else:
            print(f'Epoch {epoch_i+1}/{no_epochs}, Loss: {epoch_loss:.4f}')
    # plot losses
    plt.plot(range(no_epochs + 1), losses)    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss over Epochs')
    plt.savefig('saved/training_loss.png')
    plt.close()
    print(f'Final Loss: {losses[-1]:.4f}')
            




if __name__ == '__main__':
    no_epochs = 1000
    train_model(no_epochs)
