import argparse
import numpy as np
import torch
from os.path import isdir
import os
from models.models_pretraining import inverse_model, forward_model
from torch.utils import data
from core.functions import *
from torch import nn, optim
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

random_seed = 30
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings('ignore')

def get_data(args, test=False):

    data_dic = np.load("data/data.npy", allow_pickle=True).item()

    seismic_data = data_dic["seismic"]
    acoustic_impedance_data = data_dic["acoustic_impedance"]

    seismic_mean = torch.tensor(np.mean(seismic_data, keepdims=True)).float()
    seismic_std = torch.tensor(np.std(seismic_data, keepdims=True)).float()

    acoustic_mean= torch.tensor(np.mean(acoustic_impedance_data, keepdims=True)).float()
    acoustic_std = torch.tensor(np.std(acoustic_impedance_data, keepdims=True)).float()

    seismic_data = torch.tensor(seismic_data).float()
    acoustic_impedance_data = torch.tensor(acoustic_impedance_data).float()

    seismic_data = seismic_data.cuda()
    acoustic_impedance_data = acoustic_impedance_data.cuda()

    seismic_mean = seismic_mean.cuda()
    seismic_std = seismic_std.cuda()

    acoustic_mean = acoustic_mean.cuda()
    acoustic_std = acoustic_std.cuda()

    seismic_normalization = Normalization(mean_val=seismic_mean, std_val=seismic_std)
    acoustic_normalization = Normalization(mean_val=acoustic_mean, std_val=acoustic_std)

    seismic_data = seismic_normalization.normalize(seismic_data)
    acoustic_impedance_data = acoustic_normalization.normalize(acoustic_impedance_data)

    # expand seismic_data
    # seismic_data_fill:(N+args.width-1, 1, T), args.width has to be odd
    seismic_data_fill = torch.zeros(seismic_data.shape[0]+args.width-1, seismic_data.shape[1], seismic_data.shape[2]).cuda()
    seismic_data_fill[int((args.width-1)/2):int((args.width-1)/2)+seismic_data.shape[0], :] = seismic_data
    # seismic_data_expand:(N, 1, args.width, T)
    seismic_data_expand = torch.zeros(seismic_data.shape[0], seismic_data.shape[1], args.width, seismic_data.shape[2]).cuda()
    for i in range(seismic_data.shape[0]):
        seismic_data_expand[i, :] = seismic_data_fill[i:i+args.width, :].transpose(0, 1)
    # seismic_data:(N+args.width-1, 1, args.width+1, 470)
    seismic_data = torch.cat((seismic_data_expand, seismic_data.unsqueeze(2)), dim=2)

    if not test:
        num_samples = seismic_data.shape[0]
        indecies = np.arange(0, num_samples)
        train_indecies = indecies[(np.linspace(0, len(indecies)-1, args.num_train_wells)).astype(int)]

        train_data = data.Subset(data.TensorDataset(seismic_data, acoustic_impedance_data), train_indecies)
        train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=False)

        unlabeled_loader = data.DataLoader(data.TensorDataset(seismic_data), batch_size=args.batch_size, shuffle=True)
        return train_loader, unlabeled_loader, seismic_normalization, acoustic_normalization
    else:
        test_loader = data.DataLoader(data.TensorDataset(seismic_data, acoustic_impedance_data), batch_size=args.batch_size, shuffle=False, drop_last=False)
        return test_loader, seismic_normalization, acoustic_normalization


def get_models(args):
    if args.test_checkpoint is None:
        inverse_net = inverse_model(nonlinearity=args.nonlinearity)
        forward_net = forward_model(nonlinearity=args.nonlinearity)
        optimizer = optim.Adam(list(inverse_net.parameters())+list(forward_net.parameters()), amsgrad=True, lr=0.005)
    else:
        try:
            inverse_net = torch.load("checkpoints/" + args.test_checkpoint + "_inverse")
            forward_net = torch.load("checkpoints/" + args.test_checkpoint + "_forward")
            optimizer = torch.load("checkpoints/" + args.test_checkpoint + "_optimizer")
        except FileNotFoundError:
            try:
                inverse_net = torch.load(args.test_checkpoint + "_inverse")
                forward_net = torch.load(args.test_checkpoint + "_forward")
                optimizer = torch.load(args.test_checkpoint + "_optimizer")
            except:
                print("No checkpoint found at '{}'- Please specify the model for testing".format(args.test_checkpoint))
                exit()

    inverse_net.cuda()
    forward_net.cuda()

    return inverse_net, forward_net, optimizer


def train(args):
    train_loader, unlabeled_loader, seismic_normalization, acoustic_normalization = get_data(args)
    inverse_net, forward_net, optimizer = get_models(args)
    inverse_net.train()
    criterion = nn.MSELoss()

    # make a directory to save models if it doesn't exist
    if not isdir("checkpoints"):
        os.mkdir("checkpoints")

    loss_var = []
    print("Training the model")
    for _ in tqdm(range(args.max_epoch)):
        train_loss = []
        for x, y in train_loader:
            optimizer.zero_grad()
            x_expand = x[:, :, 0:args.width]
            x = x[:, :, args.width]
            y_pred = inverse_net(x, x_expand)
            x_rec = forward_net(y)

            property_loss = criterion(y_pred, y) + criterion(x_rec, x)
            if args.beta != 0:
                try:
                    x_u = next(unlabeled)[0]
                except:
                    unlabeled = iter(unlabeled_loader)
                    x_u = next(unlabeled)[0]

                x_u_expand = x_u[:, :, 0:args.width]
                x_u = x_u[:, :, args.width]
                y_u_pred = inverse_net(x_u, x_u_expand)
                x_u_rec = forward_net(y_u_pred)

                seismic_loss = criterion(x_u_rec, x_u)
            else:
                seismic_loss = 0
            loss = args.alpha * property_loss + args.beta * seismic_loss
            loss.backward()
            optimizer.step()

            train_loss.append(loss.detach().clone())

        train_loss = torch.mean(torch.tensor(train_loss))
        loss_var.append(train_loss)

    loss_var = torch.tensor(loss_var)
    loss_var = loss_var.cpu().numpy()
    plt.plot(np.arange(loss_var.shape[0]), loss_var)
    plt.title("Curve of Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.show()

    torch.save(inverse_net, "checkpoints/{}_inverse".format(args.session_name))
    torch.save(forward_net, "checkpoints/{}_forward".format(args.session_name))
    torch.save(optimizer, "checkpoints/{}_optimizer".format(args.session_name))


def test(args):
    test_loader, seismic_normalization, acoustic_normalization = get_data(args, test=True)
    if args.test_checkpoint is None:
        args.test_checkpoint = "checkpoints/{}".format(args.session_name)
    inverse_net, forward_net, _ = get_models(args)
    criterion = nn.MSELoss(reduction="sum")
    predicted_impedance = []
    true_impedance = []
    test_property_corr = []
    test_property_r2 = []
    inverse_net.eval()
    print("\nTesting the model\n")

    with torch.no_grad():
        test_loss = []
        for x, y in test_loader:
            x_expand = x[:, :, 0:args.width]
            x = x[:, :, args.width]
            y_pred = inverse_net(x, x_expand)
            loss = criterion(y_pred, y)/np.prod(y.shape)
            test_loss.append(loss.item())

            corr, r2 = metrics(y_pred.detach(), y.detach())
            test_property_corr.append(corr)
            test_property_r2.append(r2)

            true_impedance.append(y)
            predicted_impedance.append(y_pred)

        property_corr = torch.mean(torch.cat(test_property_corr), dim=0).squeeze()
        property_r2 = torch.mean(torch.cat(test_property_r2), dim=0).squeeze()
        loss = torch.mean(torch.tensor(test_loss))
        print("loss: {:.4f}\nCorrelation: {:0.4f}\nr2 Coeff.: {:0.4f}".format(loss, property_corr, property_r2))

        predicted_impedance = torch.cat(predicted_impedance, dim=0)
        true_impedance = torch.cat(true_impedance, dim=0)

        predicted_impedance = acoustic_normalization.unnormalize(predicted_impedance)
        true_impedance = acoustic_normalization.unnormalize(true_impedance)

        predicted_impedance = predicted_impedance.cpu()
        true_impedance = true_impedance.cpu()

        predicted_impedance = predicted_impedance.numpy()
        true_impedance = true_impedance.numpy()

        psnr = PSNR(true_impedance[:, 0], predicted_impedance[:, 0])
        ssim = SSIM(true_impedance[:, 0], predicted_impedance[:, 0])
        print("PSNR: {:.4f}\nSSIM: {:0.4f}".format(psnr, ssim))

        # plot seismic data and predicted AI
        data_dic = np.load("D:/statistics/Geostatistics/improvement/data/data.npy", allow_pickle=True).item()
        seismic_data = data_dic["seismic"]

        fig, ax = plt.subplots()
        cax = ax.imshow(20 * seismic_data[:, 0].T, cmap='rainbow', aspect=2.4, vmin=seismic_data.min(), vmax=seismic_data.max())
        ax.set_yticks([80, 160, 240, 320, 400])
        ax.set_yticklabels([0.5, 1.0, 1.5, 2.0, 2.5])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontsize(30) for label in labels]
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Trace No.", fontsize=30, labelpad=8.5)
        ax.set_ylabel("Time (s)", fontsize=30, labelpad=8.5)
        cbar = fig.colorbar(cax, shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('Seismic Wave', fontsize=25, rotation=270, labelpad=30)
        plt.show()

        fig, ax = plt.subplots()
        cax = ax.imshow(true_impedance[:, 0].T, cmap='rainbow', aspect=0.6, vmin=true_impedance.min(), vmax=true_impedance.max())
        ax.set_yticks([320, 640, 960, 1280, 1600])
        ax.set_yticklabels([0.5, 1.0, 1.5, 2.0, 2.5])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontsize(30) for label in labels]
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Trace No.", fontsize=30, labelpad=8.5)
        ax.set_ylabel("Time (s)", fontsize=30, labelpad=8.5)
        cbar = fig.colorbar(cax, shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('Impedance ((m/s)·(g/cc))', fontsize=25, rotation=270, labelpad=30)
        plt.show()

        fig, ax = plt.subplots()
        cax = ax.imshow(predicted_impedance[:, 0].T, cmap='rainbow', aspect=0.6, vmin=true_impedance.min(), vmax=true_impedance.max())
        ax.set_yticks([320, 640, 960, 1280, 1600])
        ax.set_yticklabels([0.5, 1.0, 1.5, 2.0, 2.5])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontsize(30) for label in labels]
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Trace No.", fontsize=30, labelpad=8.5)
        ax.set_ylabel("Time (s)", fontsize=30, labelpad=8.5)
        cbar = fig.colorbar(cax, shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('Impedance ((m/s)·(g/cc))', fontsize=25, rotation=270, labelpad=30)
        plt.show()

        fig, ax = plt.subplots()
        cax = ax.imshow(abs(true_impedance[:, 0].T-predicted_impedance[:, 0].T), cmap='Greys', aspect=0.6)
        ax.set_yticks([320, 640, 960, 1280, 1600])
        ax.set_yticklabels([0.5, 1.0, 1.5, 2.0, 2.5])
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontsize(30) for label in labels]
        ax.xaxis.set_label_position("top")
        ax.set_xlabel("Trace No.", fontsize=30, labelpad=8.5)
        ax.set_ylabel("Time (s)", fontsize=30, labelpad=8.5)
        cbar = fig.colorbar(cax, shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('Impedance ((m/s)·(g/cc))', fontsize=25, rotation=270, labelpad=30)
        plt.show()

        np.save("result/predicted_impedance.npy", predicted_impedance)


if __name__ == '__main__':
    # arguments and parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-width', type=int, default=5, help="Number of seismic traces in expanding data to be used for training. It must be odd")
    parser.add_argument('-num_train_wells', type=int, default=20, help="Number of AI traces from the model to be used for training")
    parser.add_argument('-max_epoch', type=int, default=1000, help="maximum number of training epochs")
    parser.add_argument('-batch_size', type=int, default=50, help="Batch size for training")
    parser.add_argument('-alpha', type=float, default=1, help="weight of property loss term")
    parser.add_argument('-beta', type=float, default=0.2, help="weight of seismic loss term")
    parser.add_argument('-index', type=int, default=480, help="plot index of the AI")
    parser.add_argument('-test_checkpoint', type=str, action="store", default=None, help="path to model to test on. When this flag is used, no training is performed")
    parser.add_argument('-session_name', type=str, action="store", default=datetime.now().strftime('%b%d_%H%M%S'), help="name of the session to be ised in saving the model")
    parser.add_argument('-nonlinearity', action="store", type=str, default="tanh", help="Type of nonlinearity for the CNN [tanh, relu]", choices=["tanh","relu"])
    parser.add_argument('-resolution_ratio', type=int, default=4, action="store",help="resolution mismtach between seismic and AI")
    args = parser.parse_args()

    # use the following command to modify the test_checkpoint:
    # parser.set_defaults(test_checkpoint="Nov19_205735")
    # args = parser.parse_args()

    if args.test_checkpoint is not None:
        test(args)
    else:
        train(args)
        test(args)


