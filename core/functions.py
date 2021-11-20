import numpy as np
import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


class Normalization:
    def __init__(self, mean_val=None, std_val=None):
        self.mean_val = mean_val
        self.std_val = std_val

    def normalize(self, x):
        return (x-self.mean_val)/self.std_val

    def unnormalize(self, x):
        return x*self.std_val + self.mean_val


def metrics(y, x):
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy()
    if torch.is_tensor(y):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()

    x_mean = np.mean(x, axis=-1, keepdims=True)
    y_mean = np.mean(y, axis=-1, keepdims=True)
    x_std = np.std(x, axis=-1, keepdims=True)
    y_std = np.std(y, axis=-1, keepdims=True)
    corr = np.mean((x-x_mean)*(y-y_mean), axis=-1, keepdims=True)/(x_std*y_std)

    S_tot = np.sum((x-x_mean)**2, axis=-1, keepdims=True)
    S_res = np.sum((x - y)**2, axis=-1, keepdims=True)

    r2 = (1-S_res/S_tot)

    return torch.tensor(corr), torch.tensor(r2)


def display_results(loss, property_corr, property_r2, args, header):
    property_corr = torch.mean(torch.cat(property_corr), dim=0).squeeze()
    property_r2 = torch.mean(torch.cat(property_r2), dim=0).squeeze()
    loss = torch.mean(torch.tensor(loss))
    print("loss: {:.4f}\nCorrelation: {:0.4f}\nr2 Coeff.  : {:0.4f}".format(loss, property_corr, property_r2))


def PSNR(true_image, real_image):
    true_image = true_image.astype(np.float64)
    real_image = real_image.astype(np.float64)

    psnr = peak_signal_noise_ratio(true_image, real_image, data_range=true_image.max()-true_image.min())
    return psnr


def SSIM(true_image, real_image):
    true_image = true_image.astype(np.float64)
    real_image = real_image.astype(np.float64)

    ssim = structural_similarity(true_image, real_image, data_range=true_image.max()-true_image.min())

    return ssim


def accuracy(predict_mean, predict_std, true_image):
    lower_bound = predict_mean - 2 * predict_std
    upper_bound = predict_mean + 2 * predict_std
    flag = (true_image > lower_bound) * (true_image < upper_bound)
    accuracy = np.sum(flag) / (flag.shape[0] * flag.shape[1])

    return accuracy
























