import torch
import torch.nn as nn
#from Data_Loaders import Data_Loaders

class Action_Conditioned_FF(nn.Module):
    def __init__(self, input_size=6, hidden_size=25, output_size=1):
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.nonlinear_activation = nn.Sigmoid()
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        hidden = self.input_to_hidden(input)
        hidden = self.nonlinear_activation(hidden)
        hidden = self.hidden_to_hidden(hidden)
        hidden = self.nonlinear_activation(hidden)
        network_output = self.hidden_to_output(hidden)
        network_output = self.nonlinear_activation(network_output)
        return network_output


    def evaluate(self, model, test_loader, loss_function):
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation
            total_loss = 0.0
            total_samples = 0
            for sample in test_loader:
                inputs = sample['input']
                labels = sample['label']
                outputs = model(inputs)
                loss = loss_function(outputs, labels.unsqueeze(1))
                total_loss += loss.item() * inputs.size(0)  # Accumulate loss
                total_samples += inputs.size(0)  # Count samples
            loss = total_loss / total_samples  # Compute average loss

        return loss
        

def main():
    model = Action_Conditioned_FF()
    

if __name__ == '__main__':
    main()
