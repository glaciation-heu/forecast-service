import torch
import torch.nn as nn
import torchbnn as bnn


class LSTM_BNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2):
        super(LSTM_BNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout,
        )
        # self.dropout = nn.Dropout(0.3)
        self.bayes_fc = bnn.BayesLinear(
            prior_mu=0.0,
            prior_sigma=0.1,
            in_features=hidden_size * 1,
            out_features=output_size,
        )
        self.bayes_sigma = bnn.BayesLinear(
            prior_mu=0,
            prior_sigma=0.1,
            in_features=hidden_size * 1,
            out_features=output_size,
        )
        # self.last_func = output_functions[featurename]
        """
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size,
        num_layers, batch_first=True, dropout=dropout)

        # Bayesian Fully Connected Layer
        self.bayes_fc = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
        in_features=hidden_size, out_features=output_size)
        """

    def forward(self, x):
        # LSTM forward pass
        # x = x.unsqueeze(-1).transpose(1, 2)
        # print(x.shape)
        # x = x.squeeze(2)
        # print(x.shape)
        # print(x.shape)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_length, hidden_size)
        # print(lstm_out.shape)
        # Get the last time step output
        lstm_out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        # print(lstm_out.shape)

        # Forward through Bayesian fully connected layer
        bnn_output = self.bayes_fc(lstm_out)  # Shape: (batch_size, output_size)
        sigma = torch.exp(self.bayes_sigma(lstm_out))
        # output = self.last_func(bnn_output)

        return bnn_output, sigma
