from torch import nn
import torch
from torch.autograd import Variable 

class Dense(nn.Module):
    def __init__(self, in_features, out_features, activation=None, using_norm=True):
        super().__init__()
        self.using_norm = using_norm
        self.linear = nn.Linear(in_features, out_features)
        if self.using_norm:
            self.norm = nn.BatchNorm1d(out_features)
        self.activation = activation

    def forward(self, x):
        if self.using_norm:
            x = self.norm(self.linear(x))
        else:
            x = self.linear(x)

        if self.activation is not None:
            x = self.activation(x)

        return x


class DenseBlock(nn.Module):
    def __init__(self, hidden_features, activation=None, using_norm=True):
        super().__init__()
        
        self.dense1 = Dense(hidden_features, hidden_features, activation, using_norm)
        self.dense2 = Dense(hidden_features, hidden_features, using_norm=using_norm)
        self.activation = activation
        
    def forward(self, x):
        y = self.dense1(x)
        y = self.dense2(y)
        out = y + x
        if self.activation is not None:
            out = self.activation(out)

        return out


class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 n_layers,
                 act=nn.LeakyReLU(0.2),
                 output_act=None,
                 using_norm=True):
        super().__init__()
        n_blocks = (n_layers - 1) // 2
        input_layer = Dense(in_features, hidden_features, act, using_norm)
        layers = [input_layer]
        for i in range(n_blocks):
            layers.append(DenseBlock(hidden_features, act, using_norm))
        output_layer = Dense(hidden_features, out_features, output_act, using_norm)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        return x
    
class MLP_new(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 n_layers,
                 act=nn.LeakyReLU(0.2),
                 output_act=None,
                 using_norm=True):
        super().__init__()
        input_layer = Dense(in_features, hidden_features[0], act, using_norm)
        layers = [input_layer]
        #print(f" Hidden features: {hidden_features}")
        
        for i in range(n_layers-2):
                layers.append(Dense(hidden_features[i], hidden_features[i+1], act, using_norm))
        layers.append(Dense(hidden_features[-1], out_features, output_act, using_norm))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        return x

class LSTM_encoder(nn.Module):
    def __init__(self, in_features, lstm_hidden_size, dim_out, num_lstm_layers=1):
        super(LSTM_encoder, self).__init__()
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        
        self.lstm = nn.LSTM(in_features, lstm_hidden_size, num_layers=num_lstm_layers, batch_first=True)
        
        # Create final layer to get desired size    
        self.dense1 = Dense(lstm_hidden_size, lstm_hidden_size , activation=nn.ELU(), using_norm=False)
        self.dense2 = Dense(lstm_hidden_size, dim_out, activation=nn.Tanh(),using_norm=False)


    def forward(self, x):  
        # Apply LSTM
        out, (hx, cx) = self.lstm(x)
        
        # Get the final hidden state of the LSTM module (from the last layer)
        out = hx[-1]
        
        # Projection
        out = self.dense1(out)
        out = self.dense2(out)

        return out

# Implementation after:
# https://medium.com/@hkabhi916/understanding-lstm-for-sequence-classification-a-practical-guide-with-pytorch-ac40e84ad3d5
# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
# https://wandb.ai/sauravmaheshkar/LSTM-PyTorch/reports/Using-LSTM-in-PyTorch-A-Tutorial-With-Examples--VmlldzoxMDA2NTA5  
class LSTM_actor(nn.Module):
    def __init__(self, in_features, 
                 hidden_features, 
                 out_features, 
                 n_layers,
                 act=nn.LeakyReLU(0.2),
                 output_act=None,
                 using_norm=True):
        
        super(LSTM_actor, self).__init__()
        self.hidden_features = hidden_features
        self.lstm = nn.LSTM(in_features, hidden_features, num_layers=1, batch_first=True)

        self.num_layers = n_layers
        n_blocks = (n_layers - 1) // 2
        input_layer = Dense(hidden_features, hidden_features, act, using_norm)
        
        layers = [input_layer]
        for i in range(n_blocks):
            layers.append(DenseBlock(hidden_features, act, using_norm))
        output_layer = Dense(hidden_features, out_features, output_act, using_norm)
        layers.append(output_layer)
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        h_0 = Variable(torch.zeros(1, x.size(0), self.hidden_features).to(x.device))
        c_0 = Variable(torch.zeros(1, x.size(0), self.hidden_features).to(x.device))
        
        # Apply LSTM
        out, (hx, cx) = self.lstm(x, (h_0, c_0))
        # print(f"Input shape: {x.shape}")
        # print(f"Output shape: {out.shape}")
        # print(f"hx shape: {hx.shape}")
        
        # Get the final hidden state of the LSTM module (from the last layer)
        #out = hx[-1]
        
        # # Projection
        out = self.model(out[:,-1,:])
    
        return out


class Discriminator(nn.Module):
    def __init__(self, in_features, hidden_features, n_layers, using_norm=True):
        super().__init__()
        n_blocks = n_layers // 2
        input_layer = Dense(in_features, hidden_features, nn.LeakyReLU(0.2), using_norm)
        layers = [input_layer]
        for i in range(n_blocks):
            layers.append(DenseBlock(hidden_features, nn.LeakyReLU(0.2), using_norm))

        output_layer = Dense(hidden_features, 1, using_norm=using_norm)
        layers.append(output_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)

        return x
