import argparse
import numpy as np
import torch
from models.models_bayes import inverse_bbb, forward_bbb
from pretraining import get_data
from core.functions import *
from torch import optim
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import scipy.stats

random_seed = 30
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings('ignore')


def get_models(args):
    if args.test_checkpoint is None:
        inverse_net = inverse_bbb(prior_var=args.prior_var, noise_ratio=args.noise_ratio)
        forward_net = forward_bbb(prior_var=args.prior_var, noise_ratio=args.noise_ratio)
        optimizer = optim.Adam(list(inverse_net.parameters())+list(forward_net.parameters()), amsgrad=True, lr=args.lr)
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
    forward_net.train()
    loss_var = []

    print("Training the model")
    for epoch in range(args.max_epoch):
        train_loss = []
        for x, y in train_loader:
            optimizer.zero_grad()
            x_expand = x[:, :, 0:args.width]
            x = x[:, :, args.width]

            property_loss = inverse_net.elbo(x, x_expand, y) + forward_net.elbo(y, x)

            if args.beta != 0:
                try:
                    x_u = next(unlabeled)[0]
                except:
                    unlabeled = iter(unlabeled_loader)
                    x_u = next(unlabeled)[0]

                x_u_expand = x_u[:, :, 0:args.width]
                x_u = x_u[:, :, args.width]
                y_u_pred = inverse_net(x_u, x_u_expand)

                seismic_loss = forward_net.elbo(y_u_pred, x_u)
            else:
                seismic_loss = 0

            loss = args.alpha * property_loss + args.beta * seismic_loss
            loss.backward(retain_graph=True)
            optimizer.step()

            train_loss.append(loss.detach().clone())

        train_loss = torch.mean(torch.tensor(train_loss))
        print("Epoch: {:}\ntrain_loss: {:0.4f}".format(epoch + 1, train_loss))
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
    num_sampling = args.num_sampling
    true_impedance = []
    inverse_net.eval()
    print("\nTesting the model")

    with torch.no_grad():
        mean = 0
        var = 0
        for t in tqdm(range(num_sampling)):
            true_impedance_t = []
            predicted_impedance_t = []
            for x, y in test_loader:
                x_expand = x[:, :, 0:args.width]
                x = x[:, :, args.width]

                y_pred = inverse_net(x, x_expand)

                true_impedance_t.append(y)
                predicted_impedance_t.append(y_pred)

            # predicted_impedance_t: (N, 1, T)
            # true_impedance_t: (N, 1, T)
            true_impedance_t = torch.cat(true_impedance_t, dim=0)
            true_impedance.append(true_impedance_t)

            # online learning for solving variance
            predicted_impedance_t = torch.cat(predicted_impedance_t, dim=0)
            predicted_impedance_t = acoustic_normalization.unnormalize(predicted_impedance_t)

            if t == 0:
                mean = predicted_impedance_t
            else:
                var = (t - 1) * var / t + torch.square(predicted_impedance_t - mean) / (t + 1)
                mean = t * mean / (t + 1) + predicted_impedance_t / (t + 1)

        # predicted_impedance_std: (N, 1, T)
        predicted_impedance_std = torch.sqrt(var)

        # true_impedance: (N, 1, T)
        true_impedance = true_impedance[0]
        true_impedance = acoustic_normalization.unnormalize(true_impedance)

        predicted_impedance_std = predicted_impedance_std.cpu()
        true_impedance = true_impedance.cpu()

        predicted_impedance_std = predicted_impedance_std.numpy()
        true_impedance = true_impedance.numpy()

        # predicted_impedance_mean: (N, 1, T)
        # it is the result of pre-train
        predicted_impedance_mean = np.load("result/predicted_impedance.npy")

        acc = accuracy(predicted_impedance_mean[:, 0], predicted_impedance_std[:, 0], true_impedance[:, 0])
        print("acc: {:0.4f}".format(acc))

        predicted_impedance_map = np.abs(predicted_impedance_mean - true_impedance)

        # plot overall uncertainty and predictive MAP
        fig, ax = plt.subplots()
        cax = ax.imshow(predicted_impedance_std[:, 0].T, cmap='rainbow', aspect=0.6, vmin=predicted_impedance_std.min(), vmax=predicted_impedance_std.max())
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
        cax = ax.imshow(predicted_impedance_map[:, 0].T, cmap='rainbow', aspect=0.6, vmin=predicted_impedance_map.min(), vmax=predicted_impedance_map.max())
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

        pcc = []
        for i in range(predicted_impedance_std.shape[0]):
            corr, _ = metrics(predicted_impedance_map[i, :], predicted_impedance_std[i, :])
            pcc.append(corr)
        pcc = torch.mean(torch.cat(pcc))
        print("pcc between predictive absolute difference and the uncertainty: {:0.4f}".format(pcc))

        n = predicted_impedance_map.shape[-1]
        t = np.sqrt(n - 2) * pcc / np.sqrt(1 - np.square(np.array(pcc)))
        print("value of t-statistics: {:0.4f}".format(torch.mean(torch.tensor(t))))
        p_value = scipy.stats.norm.cdf(t)
        print("p value of t-statistics: {:0.4f}".format(np.mean(1 - p_value)))

        # plot local uncertainty
        index_list = args.index_list
        iter = 0
        fig = plt.figure()
        for index in index_list:
            iter += 1
            ax = fig.add_subplot(1, len(index_list), iter)
            mean = predicted_impedance_mean[index, 0, :].squeeze()
            std = predicted_impedance_std[index, 0, :].squeeze()

            if iter == 1:
                ax.plot(true_impedance[index, 0, :].squeeze(), np.arange(true_impedance.shape[-1]), color="red", label="True Acoustic Impedance")
                ax.plot(mean, np.arange(true_impedance.shape[-1]), color="blue", label='Mean Posterior Prediction')
                ax.plot(np.abs(mean - true_impedance[index, 0, :]).squeeze(), np.arange(true_impedance.shape[-1]), color="green", label="Bias of the Prediction")
                ax.plot(std, np.arange(true_impedance.shape[-1]), color="yellow", label='Prediction Uncertainty', linewidth=0.5)
                ax.fill_betweenx(np.arange(true_impedance.shape[-1]), mean - 2 * std, mean + 2 * std, alpha=0.25, label='Two-Sigma Confidence Interval')
                ax.set_title("Trace No.{index}".format(index=index), fontsize=18)
                ax.invert_yaxis()
                ax.set_ylabel("Time (s)", fontsize=12)
                ax.set_xlabel("Impedance ((m/s)·(g/cc))", fontsize=12)
                ax.set_yticks([0, 320, 640, 960, 1280, 1600, 1920])
                ax.set_yticklabels([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontsize(10) for label in labels]
            else:
                ax.plot(true_impedance[index, 0, :].squeeze(), np.arange(true_impedance.shape[-1]), color="red")
                ax.plot(mean, np.arange(true_impedance.shape[-1]), color="blue")
                ax.plot(np.abs(mean - true_impedance[index, 0, :]).squeeze(), np.arange(true_impedance.shape[-1]), color="green")
                ax.plot(std, np.arange(true_impedance.shape[-1]), color="yellow", linewidth=0.5)
                ax.fill_betweenx(np.arange(true_impedance.shape[-1]), mean - 2 * std, mean + 2 * std, alpha=0.25)
                ax.set_title("Trace No.{index}".format(index=index), fontsize=18)
                ax.invert_yaxis()
                ax.set_xlabel("Impedance ((m/s)·(g/cc))", fontsize=12)
                ax.set_yticks([0, 320, 640, 960, 1280, 1600, 1920])
                ax.set_yticklabels([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
                labels = ax.get_xticklabels() + ax.get_yticklabels()
                [label.set_fontsize(10) for label in labels]

        fig.tight_layout()
        fig.subplots_adjust(top=0.95, bottom=0.2, left=0.35, right=0.65)
        fig.legend(ncol=2, bbox_to_anchor=(0.5, 0.01), loc="lower center", frameon=False, prop={'size': 15})
        fig.subplots_adjust(wspace=0.3)
        plt.show()


if __name__ == '__main__':
    # arguments and parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-width', type=int, default=5, help="Number of seismic traces in expanding data to be used for training. It must be odd")
    parser.add_argument('-num_train_wells', type=int, default=20, help="Number of AI traces from the model to be used for training")
    parser.add_argument('-max_epoch', type=int, default=3000, help="maximum number of training epochs")
    parser.add_argument('-batch_size', type=int, default=20, help="Batch size for training")
    parser.add_argument('-alpha', type=float, default=1, help="weight of property loss term")
    parser.add_argument('-beta', type=float, default=0.2, help="weight of seismic loss term")
    parser.add_argument('-index_list', type=int, default=[500, 1000], help="plot index list of the AI")
    parser.add_argument('-prior_var', type=float, default=1e-6, help="the variance of the prior distribution")
    parser.add_argument('-noise_ratio', type=float, default=1, help="the noise ratio of the likelihood distribution")
    parser.add_argument('-lr', type=float, default=0.01, help="the learning rate of Adam")
    parser.add_argument('-num_sampling', type=int, default=40, help="number of sampling for MC sampling")
    parser.add_argument('-test_checkpoint', type=str, action="store", default=None, help="path to model to test on. When this flag is used, no training is performed")
    parser.add_argument('-session_name', type=str, action="store", default=datetime.now().strftime('%b%d_%H%M%S'), help="name of the session to be ised in saving the model")
    parser.add_argument('-nonlinearity', action="store", type=str, default="tanh", help="Type of nonlinearity for the CNN [tanh, relu]", choices=["tanh","relu"])
    parser.add_argument('-resolution_ratio', type=int, default=4, action="store",help="resolution mismtach between seismic and AI")
    args = parser.parse_args()

    # use the following command to modify the test_checkpoint:
    parser.set_defaults(test_checkpoint="Nov19_213931")
    args = parser.parse_args()

    if args.test_checkpoint is not None:
        test(args)
    else:
        train(args)
        test(args)


