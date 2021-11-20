import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.functional import conv1d


class inverse_model(nn.Module):
    def __init__(self, resolution_ratio=4, nonlinearity="tanh"):
        super(inverse_model, self).__init__()
        self.resolution_ratio = resolution_ratio
        self.activation = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()

        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 5), padding=(0, 2), dilation=(1, 1))
        self.groupnorm1 = nn.GroupNorm(num_groups=1, num_channels=8)
        self.pooling1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.groupnorm2 = nn.GroupNorm(num_groups=1, num_channels=8)

        self.cnn2 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 5), padding=(0, 6), dilation=(1, 3))
        self.groupnorm3 = nn.GroupNorm(num_groups=1, num_channels=8)
        self.pooling2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.groupnorm4 = nn.GroupNorm(num_groups=1, num_channels=8)

        self.cnn3 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 5), padding=(0, 12), dilation=(1, 6))
        self.groupnorm5 = nn.GroupNorm(num_groups=1, num_channels=8)
        self.pooling3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.groupnorm6 = nn.GroupNorm(num_groups=1, num_channels=8)

        self.cnn4 = nn.Conv2d(in_channels=24, out_channels=16, kernel_size=(3, 3), padding=(0, 1))
        self.groupnorm7 = nn.GroupNorm(num_groups=1, num_channels=16)
        self.cnn5 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=(0, 1))
        self.groupnorm8 = nn.GroupNorm(num_groups=1, num_channels=16)
        self.cnn6 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=(1, 1))
        self.groupnorm9 = nn.GroupNorm(num_groups=1, num_channels=16)

        self.gru = nn.GRU(input_size=1, hidden_size=8, num_layers=3, batch_first=True, bidirectional=True)

        self.cnnt1 = nn.ConvTranspose1d(in_channels=16, out_channels=8, stride=2, kernel_size=4, padding=1)
        self.groupnorm10 = nn.GroupNorm(num_groups=1, num_channels=8)
        self.cnnt2 = nn.ConvTranspose1d(in_channels=8, out_channels=8, stride=2, kernel_size=4, padding=1)
        self.groupnorm11 = nn.GroupNorm(num_groups=1, num_channels=8)

        self.gru_out = nn.GRU(input_size=8, hidden_size=8, num_layers=1, batch_first=True, bidirectional=True)
        self.out = nn.Linear(in_features=16, out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    # shape of x:(N, 1, 470); x_expand:(N, 1, 5, 470)
    def forward(self, x, x_expand):
        cnn_out1 = self.cnn1(x_expand)
        cnn_out1 = self.groupnorm1(cnn_out1)
        cnn_out1 = self.pooling1(cnn_out1)
        cnn_out1 = self.groupnorm2(cnn_out1)

        cnn_out2 = self.cnn2(x_expand)
        cnn_out2 = self.groupnorm3(cnn_out2)
        cnn_out2 = self.pooling2(cnn_out2)
        cnn_out2 = self.groupnorm4(cnn_out2)

        cnn_out3 = self.cnn3(x_expand)
        cnn_out3 = self.groupnorm5(cnn_out3)
        cnn_out3 = self.pooling3(cnn_out3)
        cnn_out3 = self.groupnorm6(cnn_out3)

        cnn_out = torch.cat((cnn_out1, cnn_out2, cnn_out3), dim=1)
        cnn_out = self.activation(cnn_out)
        cnn_out = self.cnn4(cnn_out)
        cnn_out = self.groupnorm7(cnn_out)

        cnn_out = self.activation(cnn_out)
        cnn_out = self.cnn5(cnn_out)
        cnn_out = self.groupnorm8(cnn_out)

        cnn_out = self.activation(cnn_out)
        cnn_out = self.cnn6(cnn_out)
        cnn_out = self.groupnorm9(cnn_out)

        cnn_out = self.activation(cnn_out)
        cnn_out = cnn_out.squeeze(dim=2)

        tmp_x = x.transpose(-1, -2)
        rnn_out, _ = self.gru(tmp_x)
        rnn_out = rnn_out.transpose(-1, -2)

        x = rnn_out + cnn_out

        x = self.cnnt1(x)
        x = self.groupnorm10(x)
        x = self.activation(x)
        x = self.cnnt2(x)
        x = self.groupnorm11(x)
        x = self.activation(x)

        tmp_x = x.transpose(-1, -2)
        x, _ = self.gru_out(tmp_x)

        x = self.out(x)
        x = x.transpose(-1, -2)
        return x


class forward_model(nn.Module):
    def __init__(self,resolution_ratio=4, nonlinearity="tanh"):
        super(forward_model, self).__init__()
        self.resolution_ratio = resolution_ratio
        self.activation = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()
        self.cnn1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=9, padding=4)
        self.cnn2 = nn.Conv1d(in_channels=4, out_channels=4,kernel_size=7, padding=3)
        self.cnn3 = nn.Conv1d(in_channels=4, out_channels=1,kernel_size=3, padding=1)
        self.wavelet = nn.Conv1d(in_channels=1, out_channels=1, stride=self.resolution_ratio, kernel_size=50,padding=int((50-self.resolution_ratio+2)/2))

    def forward(self, x):
        x = self.cnn1(x)
        x = self.activation(x)
        x = self.cnn2(x)
        x = self.activation(x)
        x = self.cnn3(x)
        x = self.wavelet(x)
        return x