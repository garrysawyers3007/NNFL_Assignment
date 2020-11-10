import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, seq_len,
                 num_layers, batch_size, batch_first=False):
        
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=batch_first, bidirectional=False)
        # Output of the LSTM is of shape(seq_len, batch, num_directions*hidden_size)
        # Therefore flattening it, it would become (batch, seq_len*num_directions*hidden_size)
        # This example assumes bidirectional is false, change accordingly if True

        self.linear = nn.Linear(in_features=seq_len * 1 * hidden_size, out_features=num_classes)

        self.num_layers = num_layers
        self.batch = batch_size
        self.hidden = hidden_size

    def forward(self, x):
        # input shape is (seq_len, batch, features)
        h0, c0 = torch.randn(self.num_layers, self.batch, self.hidden), torch.randn(self.num_layers, self.batch,
                                                                                    self.hidden)  # 2*self.hidden is bidirectional

        output, (hn, cn) = self.lstm(x, (h0, c0))
        # output is of shape (seq_len, batch, direction*hidden)
        output = output.permute(1, 0, 2) # (batch, seq_len, direction*hidden)
        output = output.view(output.size(0), -1)

        return self.linear(output)