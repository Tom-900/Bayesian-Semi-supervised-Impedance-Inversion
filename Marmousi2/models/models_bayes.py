import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from torch.autograd import Variable

inverse_net = torch.load(".../Marmousi2/checkpoints/Nov19_205735_inverse")
forward_net = torch.load(".../Marmousi2/checkpoints/Nov19_205735_forward")

class inverse_bbb(nn.Module):
    def __init__(self, prior_var=10, noise_ratio=0.1):
        super(inverse_bbb, self).__init__()
        self.KL = 0
        self.prior_var = prior_var
        self.noise_ratio = noise_ratio

        self.rho_gru_weight = nn.Parameter(torch.zeros((16, 1)), requires_grad=True)
        self.rho_gru_bias = nn.Parameter(torch.zeros(16), requires_grad=True)
        self.rho_gru_out_weight = nn.Parameter(torch.zeros((16, 8)), requires_grad=True)
        self.rho_gru_out_bias = nn.Parameter(torch.zeros(16), requires_grad=True)

        for k, v in inverse_net.named_parameters():
            name = "rho_{}".format(k).replace(".", "_")
            params = nn.Parameter(torch.zeros(v.shape), requires_grad=True)
            setattr(self, name, params)

    def rho_to_var(self, rho):
        return torch.square(torch.log(1 + torch.exp(rho)))

    def KL_divergence(self, mu1, mu2, var1, var2):
        kl = - torch.sum(torch.log(var1 / var2) - var1 / var2 - torch.square(mu1 - mu2) / var2 + 1) / 2
        return kl

    def linear_var(self, rho_weight, rho_bias, inverse_weight, inverse_bias):
        var_weight = self.rho_to_var(rho_weight)
        var_bias = self.rho_to_var(rho_bias)
        kl_weight = self.KL_divergence(inverse_weight, torch.zeros_like(inverse_weight), var_weight, self.prior_var)
        kl_bias = self.KL_divergence(inverse_bias, torch.zeros_like(inverse_bias), var_bias, self.prior_var)
        return var_weight, var_bias, kl_weight, kl_bias

    def forward(self, x, x_expand):

         # update nn.sequential cnn1
        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_cnn1_weight, self.rho_cnn1_bias, inverse_net.cnn1.weight, inverse_net.cnn1.bias)

        self.rho_cnn1_weight_kl = kl_weight
        self.rho_cnn1_bias_kl = kl_bias

        var = F.conv2d(x_expand.square(), var_weight, var_bias, padding=(0, 2), dilation=(1, 1))
        cnn_out1 = inverse_net.cnn1(x_expand)
        epsilon = Normal(0, 1).sample(cnn_out1.shape).cuda()
        cnn_out1 = cnn_out1 + epsilon * var.sqrt()

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_groupnorm1_weight, self.rho_groupnorm1_bias, inverse_net.groupnorm1.weight, inverse_net.groupnorm1.bias)
        self.rho_groupnorm1_weight_kl = kl_weight
        self.rho_groupnorm1_bias_kl = kl_bias

        var = F.group_norm(cnn_out1.square(), 1, var_weight, var_bias)
        cnn_out1 = inverse_net.groupnorm1(cnn_out1)
        epsilon = Normal(0, 1).sample(cnn_out1.shape).cuda()
        cnn_out1 = cnn_out1 + epsilon * var.sqrt()

        cnn_out1 = inverse_net.pooling1(cnn_out1)

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_groupnorm2_weight, self.rho_groupnorm2_bias, inverse_net.groupnorm2.weight, inverse_net.groupnorm2.bias)
        self.rho_groupnorm2_weight_kl = kl_weight
        self.rho_groupnorm2_bias_kl = kl_bias

        var = F.group_norm(cnn_out1.square(), 1, var_weight, var_bias)
        cnn_out1 = inverse_net.groupnorm2(cnn_out1)
        epsilon = Normal(0, 1).sample(cnn_out1.shape).cuda()
        cnn_out1 = cnn_out1 + epsilon * var.sqrt()

         # update Sequential cnn2
        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_cnn2_weight, self.rho_cnn2_bias, inverse_net.cnn2.weight, inverse_net.cnn2.bias)

        self.rho_cnn2_weight_kl = kl_weight
        self.rho_cnn2_bias_kl = kl_bias

        var = F.conv2d(x_expand.square(), var_weight, var_bias, padding=(0, 6), dilation=(1, 3))
        cnn_out2 = inverse_net.cnn2(x_expand)
        epsilon = Normal(0, 1).sample(cnn_out2.shape).cuda()
        cnn_out2 = cnn_out2 + epsilon * var.sqrt()

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_groupnorm3_weight, self.rho_groupnorm3_bias, inverse_net.groupnorm3.weight, inverse_net.groupnorm3.bias)
        self.rho_groupnorm3_weight_kl = kl_weight
        self.rho_groupnorm3_bias_kl = kl_bias

        var = F.group_norm(cnn_out2.square(), 1, var_weight, var_bias)
        cnn_out2 = inverse_net.groupnorm3(cnn_out2)
        epsilon = Normal(0, 1).sample(cnn_out2.shape).cuda()
        cnn_out2 = cnn_out2 + epsilon * var.sqrt()

        cnn_out2 = inverse_net.pooling2(cnn_out2)

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_groupnorm4_weight, self.rho_groupnorm4_bias, inverse_net.groupnorm4.weight, inverse_net.groupnorm4.bias)
        self.rho_groupnorm4_weight_kl = kl_weight
        self.rho_groupnorm4_bias_kl = kl_bias

        var = F.group_norm(cnn_out2.square(), 1, var_weight, var_bias)
        cnn_out2 = inverse_net.groupnorm4(cnn_out2)
        epsilon = Normal(0, 1).sample(cnn_out2.shape).cuda()
        cnn_out2 = cnn_out2 + epsilon * var.sqrt()

         # update Sequential cnn3
        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_cnn3_weight, self.rho_cnn3_bias, inverse_net.cnn3.weight, inverse_net.cnn3.bias)

        self.rho_cnn3_weight_kl = kl_weight
        self.rho_cnn3_bias_kl = kl_bias

        var = F.conv2d(x_expand.square(), var_weight, var_bias, padding=(0, 12), dilation=(1, 6))
        cnn_out3 = inverse_net.cnn3(x_expand)
        epsilon = Normal(0, 1).sample(cnn_out3.shape).cuda()
        cnn_out3 = cnn_out3 + epsilon * var.sqrt()

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_groupnorm5_weight, self.rho_groupnorm5_bias, inverse_net.groupnorm5.weight, inverse_net.groupnorm5.bias)
        self.rho_groupnorm5_weight_kl = kl_weight
        self.rho_groupnorm5_bias_kl = kl_bias

        var = F.group_norm(cnn_out3.square(), 1, var_weight, var_bias)
        cnn_out3 = inverse_net.groupnorm5(cnn_out3)
        epsilon = Normal(0, 1).sample(cnn_out3.shape).cuda()
        cnn_out3 = cnn_out3 + epsilon * var.sqrt()

        cnn_out3 = inverse_net.pooling3(cnn_out3)

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_groupnorm6_weight, self.rho_groupnorm6_bias, inverse_net.groupnorm6.weight, inverse_net.groupnorm6.bias)
        self.rho_groupnorm6_weight_kl = kl_weight
        self.rho_groupnorm6_bias_kl = kl_bias

        var = F.group_norm(cnn_out3.square(), 1, var_weight, var_bias)
        cnn_out3 = inverse_net.groupnorm6(cnn_out3)
        epsilon = Normal(0, 1).sample(cnn_out3.shape).cuda()
        cnn_out3 = cnn_out3 + epsilon * var.sqrt()

        cnn_out = torch.cat((cnn_out1, cnn_out2, cnn_out3), dim=1)

         # update cnn4 and groupnorm
        cnn_out = inverse_net.activation(cnn_out)

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_cnn4_weight, self.rho_cnn4_bias, inverse_net.cnn4.weight, inverse_net.cnn4.bias)

        self.rho_cnn4_weight_kl = kl_weight
        self.rho_cnn4_bias_kl = kl_bias

        var = F.conv2d(cnn_out.square(), var_weight, var_bias, padding=(0, 1))
        cnn_out = inverse_net.cnn4(cnn_out)
        epsilon = Normal(0, 1).sample(cnn_out.shape).cuda()
        cnn_out = cnn_out + epsilon * var.sqrt()

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_groupnorm7_weight, self.rho_groupnorm7_bias, inverse_net.groupnorm7.weight, inverse_net.groupnorm7.bias)
        self.rho_groupnorm7_weight_kl = kl_weight
        self.rho_groupnorm7_bias_kl = kl_bias

        var = F.group_norm(cnn_out.square(), 1, var_weight, var_bias)
        cnn_out = inverse_net.groupnorm7(cnn_out)
        epsilon = Normal(0, 1).sample(cnn_out.shape).cuda()
        cnn_out = cnn_out + epsilon * var.sqrt()

         # Update cnn5 and groupnorm
        cnn_out = inverse_net.activation(cnn_out)

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_cnn5_weight, self.rho_cnn5_bias, inverse_net.cnn5.weight, inverse_net.cnn5.bias)

        self.rho_cnn5_weight_kl = kl_weight
        self.rho_cnn5_bias_kl = kl_bias

        var = F.conv1d(cnn_out.square(), var_weight, var_bias, padding=(0, 1))
        cnn_out = inverse_net.cnn5(cnn_out)
        epsilon = Normal(0, 1).sample(cnn_out.shape).cuda()
        cnn_out = cnn_out + epsilon * var.sqrt()

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_groupnorm8_weight, self.rho_groupnorm8_bias, inverse_net.groupnorm8.weight, inverse_net.groupnorm8.bias)
        self.rho_groupnorm8_weight_kl = kl_weight
        self.rho_groupnorm8_bias_kl = kl_bias

        var = F.group_norm(cnn_out.square(), 1, var_weight, var_bias)
        cnn_out = inverse_net.groupnorm8(cnn_out)
        epsilon = Normal(0, 1).sample(cnn_out.shape).cuda()
        cnn_out = cnn_out + epsilon * var.sqrt()

        # Update cnn6 and groupnorm
        cnn_out = inverse_net.activation(cnn_out)

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_cnn6_weight, self.rho_cnn6_bias, inverse_net.cnn6.weight, inverse_net.cnn6.bias)

        self.rho_cnn6_weight_kl = kl_weight
        self.rho_cnn6_bias_kl = kl_bias

        var = F.conv1d(cnn_out.square(), var_weight, var_bias)
        cnn_out = inverse_net.cnn6(cnn_out)
        epsilon = Normal(0, 1).sample(cnn_out.shape).cuda()
        cnn_out = cnn_out + epsilon * var.sqrt()

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_groupnorm9_weight, self.rho_groupnorm9_bias, inverse_net.groupnorm9.weight, inverse_net.groupnorm9.bias)
        self.rho_groupnorm9_weight_kl = kl_weight
        self.rho_groupnorm9_bias_kl = kl_bias

        var = F.group_norm(cnn_out.square(), 1, var_weight, var_bias)
        cnn_out = inverse_net.groupnorm9(cnn_out)
        epsilon = Normal(0, 1).sample(cnn_out.shape).cuda()
        cnn_out = cnn_out + epsilon * var.sqrt()

        cnn_out = inverse_net.activation(cnn_out)
        cnn_out = cnn_out.squeeze(dim=2)

        # Update gru
        tmp_x = x.transpose(-1, -2)

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_gru_weight, self.rho_gru_bias, torch.zeros(1).cuda(), torch.zeros(1).cuda())

        self.rho_gru_weight_kl = kl_weight
        self.rho_gru_bias_kl = kl_bias

        var = F.linear(tmp_x.square(), var_weight, var_bias)

        rnn_out, _ = inverse_net.gru(tmp_x)
        epsilon = Normal(0, 1).sample(rnn_out.shape).cuda()
        rnn_out = rnn_out + epsilon * var.sqrt()

        rnn_out = rnn_out.transpose(-1, -2)

        x = rnn_out + cnn_out

        # Update upsamping
        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_cnnt1_weight, self.rho_cnnt1_bias, inverse_net.cnnt1.weight, inverse_net.cnnt1.bias)

        self.rho_cnnt1_weight_kl = kl_weight
        self.rho_cnnt1_bias_kl = kl_bias

        var = F.conv_transpose1d(x.square(), var_weight, var_bias, stride=2, padding=1)
        x = inverse_net.cnnt1(x)
        epsilon = Normal(0, 1).sample(x.shape).cuda()
        x = x + epsilon * var.sqrt()

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_groupnorm10_weight, self.rho_groupnorm10_bias, inverse_net.groupnorm10.weight, inverse_net.groupnorm10.bias)
        self.rho_groupnorm10_weight_kl = kl_weight
        self.rho_groupnorm10_bias_kl = kl_bias

        var = F.group_norm(x.square(), 1, var_weight, var_bias)
        x = inverse_net.groupnorm10(x)
        epsilon = Normal(0, 1).sample(x.shape).cuda()
        x = x + epsilon * var.sqrt()

        x = inverse_net.activation(x)

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_cnnt2_weight, self.rho_cnnt2_bias, inverse_net.cnnt2.weight, inverse_net.cnnt2.bias)

        self.rho_cnnt2_weight_kl = kl_weight
        self.rho_cnnt2_bias_kl = kl_bias

        var = F.conv_transpose1d(x.square(), var_weight, var_bias, stride=2, padding=1)
        x = inverse_net.cnnt2(x)
        epsilon = Normal(0, 1).sample(x.shape).cuda()
        x = x + epsilon * var.sqrt()

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_groupnorm11_weight, self.rho_groupnorm11_bias, inverse_net.groupnorm11.weight, inverse_net.groupnorm11.bias)
        self.rho_groupnorm11_weight_kl = kl_weight
        self.rho_groupnorm11_bias_kl = kl_bias

        var = F.group_norm(x.square(), 1, var_weight, var_bias)
        x = inverse_net.groupnorm11(x)
        epsilon = Normal(0, 1).sample(x.shape).cuda()
        x = x + epsilon * var.sqrt()

        x = inverse_net.activation(x)

        # update gru_out
        tmp_x = x.transpose(-1, -2)

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_gru_out_weight, self.rho_gru_bias, torch.zeros(1).cuda(), torch.zeros(1).cuda())

        self.rho_gru_out_weight_kl = kl_weight
        self.rho_gru_out_bias_kl = kl_bias

        var = F.linear(tmp_x.square(), var_weight, var_bias)
        x, _ = inverse_net.gru_out(tmp_x)
        epsilon = Normal(0, 1).sample(x.shape).cuda()
        x = x + epsilon * var.sqrt()

        # update out
        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_out_weight, self.rho_out_bias, inverse_net.out.weight, inverse_net.out.bias)
        self.rho_out_weight_kl = kl_weight
        self.rho_out_bias_kl = kl_bias

        var = F.linear(x.square(), var_weight, var_bias)
        x = inverse_net.out(x)
        epsilon = Normal(0, 1).sample(x.shape).cuda()
        x = x + epsilon * var.sqrt()

        x = x.transpose(-1, -2)
        return x

    def elbo(self, x, x_expand, target):

        y = self.forward(x, x_expand)
        kl_divergence = 0
        for k, v in inverse_net.named_parameters():
            try:
                name = "rho_{}".format(k).replace(".", "_")
                name = name + "_kl"
                kl_divergence = kl_divergence + getattr(self, name)
            except:
                pass
                continue

        kl_divergence = kl_divergence + self.rho_gru_weight_kl
        kl_divergence = kl_divergence + self.rho_gru_bias_kl
        kl_divergence = kl_divergence + self.rho_gru_out_weight_kl
        kl_divergence = kl_divergence + self.rho_gru_out_bias_kl

        criterion = nn.MSELoss(reduction="sum")
        loss = criterion(y, target)

        elbo = kl_divergence + self.noise_ratio * loss
        return elbo


class forward_bbb(nn.Module):
    def __init__(self, prior_var=10, noise_ratio=0.1):
        super(forward_bbb, self).__init__()
        self.KL = 0
        self.prior_var = prior_var
        self.noise_ratio = noise_ratio
        for k, v in forward_net.named_parameters():
            name = "rho_{}".format(k).replace(".", "_")
            params = nn.Parameter(torch.zeros(v.shape), requires_grad=True)
            setattr(self, name, params)

    def rho_to_var(self, rho):
        return torch.square(torch.log(1 + torch.exp(rho)))

    def KL_divergence(self, mu1, mu2, var1, var2):
        kl = - torch.sum(torch.log(var1 / var2) - var1 / var2 - torch.square(mu1 - mu2) / var2 + 1) / 2
        return kl

    def linear_var(self, rho_weight, rho_bias, inverse_weight, inverse_bias):
        var_weight = self.rho_to_var(rho_weight)
        var_bias = self.rho_to_var(rho_bias)
        kl_weight = self.KL_divergence(inverse_weight, torch.zeros_like(inverse_weight), var_weight, self.prior_var)
        kl_bias = self.KL_divergence(inverse_bias, torch.zeros_like(inverse_bias), var_bias, self.prior_var)
        return var_weight, var_bias, kl_weight, kl_bias

    def forward(self, x):
        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_cnn1_weight, self.rho_cnn1_bias, forward_net.cnn1.weight, forward_net.cnn1.bias)

        self.rho_cnn1_weight_kl = kl_weight
        self.rho_cnn1_bias_kl = kl_bias

        var = F.conv1d(x.square(), var_weight, var_bias, padding=4)
        x = forward_net.cnn1(x)
        epsilon = Normal(0, 1).sample(x.shape).cuda()
        x = x + epsilon * var.sqrt()

        x = forward_net.activation(x)

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_cnn2_weight, self.rho_cnn2_bias, forward_net.cnn2.weight, forward_net.cnn2.bias)

        self.rho_cnn2_weight_kl = kl_weight
        self.rho_cnn2_bias_kl = kl_bias

        var = F.conv1d(x.square(), var_weight, var_bias, padding=3)
        x = forward_net.cnn2(x)
        epsilon = Normal(0, 1).sample(x.shape).cuda()
        x = x + epsilon * var.sqrt()

        x = forward_net.activation(x)

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_cnn3_weight, self.rho_cnn3_bias, forward_net.cnn3.weight, forward_net.cnn3.bias)

        self.rho_cnn3_weight_kl = kl_weight
        self.rho_cnn3_bias_kl = kl_bias

        var = F.conv1d(x.square(), var_weight, var_bias, padding=1)
        x = forward_net.cnn3(x)
        epsilon = Normal(0, 1).sample(x.shape).cuda()
        x = x + epsilon * var.sqrt()

        var_weight, var_bias, kl_weight, kl_bias = self.linear_var(self.rho_wavelet_weight, self.rho_wavelet_bias, forward_net.wavelet.weight, forward_net.wavelet.bias)

        self.rho_wavelet_weight_kl = kl_weight
        self.rho_wavelet_bias_kl = kl_bias

        var = F.conv1d(x.square(), var_weight, var_bias, stride=forward_net.resolution_ratio, padding=int((50-forward_net.resolution_ratio+2)/2))
        x = forward_net.wavelet(x)
        epsilon = Normal(0, 1).sample(x.shape).cuda()
        x = x + epsilon * var.sqrt()

        return x

    def elbo(self, x, target):
        kl_divergence = 0
        for k, v in inverse_net.named_parameters():
            try:
                name = "rho_{}".format(k).replace(".", "_")
                name = name + "_kl"
                kl_divergence = kl_divergence + getattr(self, name)
            except:
                pass
                continue

        y = self.forward(x)
        criterion = nn.MSELoss(reduction="sum")
        loss = criterion(y, target)

        elbo = kl_divergence + self.noise_ratio * loss
        return elbo

