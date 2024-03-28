import torch
from torch import nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Only take the output from the final timetep
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def flatten(self):
        self.lstm.flatten_parameters()


class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(TransformerEncoderModel, self).__init__()
        self.encode = nn.Linear(input_dim, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout),
            num_layers=num_layers
        )
        self.decode = nn.Linear(d_model, 1)

    def forward(self, src):
        input_encoded = self.encode(src)
        encoded = self.transformer_encoder(input_encoded)
        output = self.decode(encoded[:, -1, :])
        return output

    def seq_forward(self, src):
        input_encoded = self.encode(src)
        encoded = self.transformer_encoder(input_encoded)
        output = self.decode(encoded[:, -1, :])
        return output

    def flatten(self):
        pass