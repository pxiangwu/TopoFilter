import torch.nn as nn
import torch.nn.parallel
import random
import argparse
from network.resnet import resnet18, resnet34
from network.pointnet import PointNetCls
from torch.utils.data import DataLoader
import os
import numpy as np
from data.cifar10_train_val_test import CIFAR10, CIFAR100
from data.modelnet40 import ModelNet40
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from termcolor import cprint
from knn_utils import calc_knn_graph, calc_topo_weights_with_components_idx
from noise import noisify_with_P, noisify_cifar10_asymmetric, \
    noisify_cifar100_asymmetric, noisify_pairflip, noisify_modelnet40_asymmetric
import copy
from scipy.stats import mode


def check_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


# random seed related
def _init_fn(worker_id):
    np.random.seed(77 + worker_id)


def main(args):
    random_seed = args.seed

    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True  # need to set to True as well

    print('Using {}\nTest on {}\nRandom Seed {}\nk_cc {}\nk_outlier {}\nevery n epoch {}\n'.format(args.net,
                                                                                      args.dataset, 
                                                                                      args.seed,
                                                                                      args.k_cc,
                                                                                      args.k_outlier,
                                                                                      args.every))

    # -- training parameters --
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        num_epoch = 180
        milestone = [int(x) for x in args.milestone.split(',')]
        batch_size = 128
    elif args.dataset == 'pc':
        num_epoch = 90
        milestone = [30, 60]
        batch_size = 128
    else:
        ValueError('Invalid Dataset!')

    start_epoch = 0
    num_workers = args.nworker

    weight_decay = 1e-4
    gamma = 0.5
    lr = 0.001

    which_data_set = args.dataset  # 'cifar100', 'cifar10', 'pc'
    noise_level = args.noise  # noise level
    noise_type = args.type  # "uniform", "asymmetric"

    train_val_ratio = 0.8
    which_net = args.net  # "resnet34", "resnet18", "pc"

    # -- denoising related parameters --
    k_cc = args.k_cc
    k_outlier = args.k_outlier
    when_to_denoise = args.start_clean           # starting from which epoch we denoise
    denoise_every_n_epoch = args.every       # every n epoch, we perform denoising

    # -- specify dataset --
    # data augmentation
    if which_data_set[:5] == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = None
        transform_test = None

    if which_data_set == 'cifar10':
        trainset = CIFAR10(root='./data', split='train', train_ratio=train_val_ratio, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                  worker_init_fn=_init_fn)

        valset = CIFAR10(root='./data', split='val', train_ratio=train_val_ratio, download=True, transform=transform_test)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        testset = CIFAR10(root='./data', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        num_class = 10
        in_channel = 3
    elif which_data_set == 'cifar100':
        trainset = CIFAR100(root='./data', split='train', train_ratio=train_val_ratio, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                  worker_init_fn=_init_fn)

        valset = CIFAR100(root='./data', split='val', train_ratio=train_val_ratio, download=True, transform=transform_test)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        testset = CIFAR100(root='./data', split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        num_class = 100
        in_channel = 3
    elif which_data_set == 'pc':
        trainset = ModelNet40(split='train', train_ratio=train_val_ratio, num_ptrs=1024, random_jitter=True)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                                  worker_init_fn=_init_fn, drop_last=True)

        valset = ModelNet40(split='val', train_ratio=train_val_ratio, num_ptrs=1024)
        valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        testset = ModelNet40(split='test', num_ptrs=1024)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        num_class = 40
    else:
        ValueError('Invalid Dataset!')

    print('train data size:', len(trainset))
    print('validation data size:', len(valset))
    print('test data size:', len(testset))
    ntrain = len(trainset)

    # -- generate noise --
    y_train = trainset.get_data_labels()
    y_train = np.array(y_train)

    noise_y_train = None
    keep_indices = None
    p = None

    if noise_type == 'none':
        pass
    else:
        if noise_type == "uniform":
            noise_y_train, p, keep_indices = noisify_with_P(y_train, nb_classes=num_class, noise=noise_level, random_state=random_seed)
            trainset.update_corrupted_label(noise_y_train)
            print("apply uniform noise")
        else:
            if which_data_set == 'cifar10':
                noise_y_train, p, keep_indices = noisify_cifar10_asymmetric(y_train, noise=noise_level, random_state=random_seed)
            elif which_data_set == 'cifar100':
                noise_y_train, p, keep_indices = noisify_cifar100_asymmetric(y_train, noise=noise_level, random_state=random_seed)
            elif which_data_set == 'pc':
                noise_y_train, p, keep_indices = noisify_modelnet40_asymmetric(y_train, noise=noise_level,
                                                                               random_state=random_seed)

            trainset.update_corrupted_label(noise_y_train)
            print("apply asymmetric noise")
        print("clean data num:", len(keep_indices))
        print("probability transition matrix:\n{}".format(p))

    # -- create log file --
    file_name = '(' + which_data_set + '_' + which_net + ')' \
                + 'type_' + noise_type + '_noise_' + str(noise_level) \
                + '_k_cc_' + str(k_cc) + '_k_outlier_' + str(k_outlier) + '_start_' + str(when_to_denoise) \
                + '_every_' + str(denoise_every_n_epoch) + '.txt'
    log_dir = check_folder('logs/logs_txt_' + str(random_seed))
    file_name = os.path.join(log_dir, file_name)
    saver = open(file_name, "w")

    saver.write('noise type: {}\nnoise level: {}\nk_cc: {}\nk_outlier: {}\nwhen_to_apply_epoch: {}\n'.format(
        noise_type, noise_level, k_cc, k_outlier, when_to_denoise))
    if noise_type != 'none':
        saver.write('total clean data num: {}\n'.format(len(keep_indices)))
        saver.write('probability transition matrix:\n{}\n'.format(p))
    saver.flush()

    # -- set network, optimizer, scheduler, etc --
    if which_net == 'resnet18':
        net = resnet18(in_channel=in_channel, num_classes=num_class)
        feature_size = 512
    elif which_net == 'resnet34':
        net = resnet34(in_channel=in_channel, num_classes=num_class)
        feature_size = 512
    elif which_net == 'pc':
        net = PointNetCls(k=num_class)
        feature_size = 256
    else:
        ValueError('Invalid network!')

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    ################################################
    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=gamma)

    criterion = nn.NLLLoss()  # since the output of network is by log softmax

    # -- misc --
    best_acc = 0
    best_epoch = 0
    best_weights = None

    curr_trainloader = trainloader

    big_comp = set()

    patience = args.patience
    no_improve_counter = 0

    # -- start training --
    for epoch in range(start_epoch, num_epoch):
        train_correct = 0
        train_loss = 0
        train_total = 0

        exp_lr_scheduler.step()
        net.train()
        print("current train data size:", len(curr_trainloader.dataset))

        for _, (images, labels, _) in enumerate(curr_trainloader):
            if images.size(0) == 1:  # when batch size equals 1, skip, due to batch normalization
                continue
            images, labels = images.to(device), labels.to(device)

            outputs, features = net(images)
            log_outputs = torch.log_softmax(outputs, 1)

            loss = criterion(log_outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = train_correct / train_total * 100.

        cprint('epoch: {}'.format(epoch), 'white')
        cprint('train accuracy: {}\ntrain loss:{}'.format(train_acc, train_loss), 'yellow')

        # --- compute big connected components ---
        net.eval()
        features_all = torch.zeros(ntrain, feature_size).to(device)
        prob_all = torch.zeros(ntrain, num_class)

        labels_all = torch.zeros(ntrain, num_class)
        train_gt_labels = np.zeros(ntrain, dtype=np.uint64)
        train_pred_labels = np.zeros(ntrain, dtype=np.uint64)

        for _, (images, labels, indices) in enumerate(trainloader):
            images = images.to(device)

            outputs, features = net(images)

            softmax_outputs = torch.softmax(outputs, 1)

            features_all[indices] = features.detach()
            prob_all[indices] = softmax_outputs.detach().cpu()

            tmp_zeros = torch.zeros(labels.shape[0], num_class)
            tmp_zeros[torch.arange(labels.shape[0]), labels] = 1.0
            labels_all[indices] = tmp_zeros

            train_gt_labels[indices] = labels.cpu().numpy().astype(np.int64)
            train_pred_labels[indices] = labels.cpu().numpy().astype(np.int64)

        if epoch >= when_to_denoise and (epoch - when_to_denoise) % denoise_every_n_epoch == 0:
            cprint('\n>> Computing Big Components <<', 'white')

            labels_all = labels_all.numpy()
            train_gt_labels = train_gt_labels.tolist()
            train_pred_labels = np.squeeze(train_pred_labels).ravel().tolist()

            _, idx_of_comp_idx2 = calc_topo_weights_with_components_idx(ntrain, labels_all, features_all,
                                                                        train_gt_labels, train_pred_labels, k=k_cc,
                                                                        use_log=False, cp_opt=3, nclass=num_class)

            # --- update largest connected component ---
            cur_big_comp = list(set(range(ntrain)) - set(idx_of_comp_idx2))
            big_comp = big_comp.union(set(cur_big_comp))

            # --- remove outliers in largest connected component ---
            big_com_idx = list(big_comp)
            feats_big_comp = features_all[big_com_idx]
            labels_big_comp = np.array(train_gt_labels)[big_com_idx]

            knnG_list = calc_knn_graph(feats_big_comp, k=args.k_outlier)

            knnG_list = np.array(knnG_list)
            knnG_shape = knnG_list.shape
            knn_labels = labels_big_comp[knnG_list.ravel()]
            knn_labels = np.reshape(knn_labels, knnG_shape)

            majority, counts = mode(knn_labels, axis=-1)
            majority = majority.ravel()
            counts = counts.ravel()

            if args.zeta > 1.0:  # use majority vote
                non_outlier_idx = np.where(majority == labels_big_comp)[0]
                outlier_idx = np.where(majority != labels_big_comp)[0]
                outlier_idx = np.array(list(big_comp))[outlier_idx]
                print(">> majority == labels_big_comp -> size: ", len(non_outlier_idx))

            else:  # zeta filtering
                non_outlier_idx = np.where((majority == labels_big_comp) & (counts >= args.k_outlier * args.zeta))[0]
                print(">> zeta {}, then non_outlier_idx -> size: {}".format(args.zeta, len(non_outlier_idx)))

                outlier_idx = np.where(majority != labels_big_comp)[0]
                outlier_idx = np.array(list(big_comp))[outlier_idx]

            cprint(">> The number of outliers: {}".format(len(outlier_idx)), 'red')
            cprint(">> The purity of outliers: {}".format(np.sum(y_train[outlier_idx] == noise_y_train[outlier_idx])
                                                            / float(len(outlier_idx))), 'red')

            big_comp = np.array(list(big_comp))[non_outlier_idx]
            big_comp = set(big_comp.tolist())

            # --- construct updated dataset set, which contains the collected clean data ---
            if which_data_set == 'cifar10':
                trainset_ignore_noisy_data = CIFAR10(root='./data', split='train', train_ratio=train_val_ratio,
                                                     download=True, transform=transform_train)
            elif which_data_set == 'cifar100':
                trainset_ignore_noisy_data = CIFAR100(root='./data', split='train', train_ratio=train_val_ratio,
                                                      download=True, transform=transform_train)
            else:
                trainset_ignore_noisy_data = ModelNet40(split='train', train_ratio=train_val_ratio,
                                                        num_ptrs=1024, random_jitter=True)

            trainloader_ignore_noisy_data = torch.utils.data.DataLoader(trainset_ignore_noisy_data,
                                                                        batch_size=batch_size,
                                                                        shuffle=True, num_workers=num_workers,
                                                                        worker_init_fn=_init_fn, drop_last=True)
            curr_trainloader = trainloader_ignore_noisy_data

            noisy_data_indices = list(set(range(ntrain)) - big_comp)

            trainset_ignore_noisy_data.update_corrupted_label(noise_y_train)
            trainset_ignore_noisy_data.ignore_noise_data(noisy_data_indices)

            clean_data_num = len(big_comp.intersection(set(keep_indices)))
            noise_data_num = len(big_comp) - clean_data_num
            print("Big Comp Number:", len(big_comp))
            print("Found Noisy Data Number:", noise_data_num)
            print("Found True Data Number:", clean_data_num)

            # compute purity of the component
            cc_size = len(big_comp)
            equal = np.sum(noise_y_train[list(big_comp)] == y_train[list(big_comp)])
            ratio = equal / float(cc_size)
            print("Purity of current component: {}".format(ratio))

            noise_size = len(noisy_data_indices)
            equal = np.sum(noise_y_train[noisy_data_indices] == y_train[noisy_data_indices])
            print("Purity of data outside component: {}".format(equal / float(noise_size)))

            saver.write('Purity {}\tsize{}\t'.format(ratio, cc_size))

        # --- validation ---
        val_total = 0
        val_correct = 0
        net.eval()
        with torch.no_grad():
            for _, (images, labels, _) in enumerate(valloader):
                images, labels = images.to(device), labels.to(device)

                outputs, _ = net(images)

                val_total += images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total * 100.

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(net.state_dict())
            no_improve_counter = 0
        else:
            no_improve_counter += 1
            if no_improve_counter >= patience:
                print('>> No improvement for {} epochs. Stop at epoch {}'.format(patience, epoch))
                saver.write('>> No improvement for {} epochs. Stop at epoch {}'.format(patience, epoch))
                saver.write('>> val epoch: {}\n>> current accuracy: {}\n'.format(epoch, val_acc))
                saver.write('>> best accuracy: {}\tbest epoch: {}\n\n'.format(best_acc, best_epoch))
                break

        cprint('val accuracy: {}'.format(val_acc), 'cyan')
        cprint('>> best accuracy: {}\n>> best epoch: {}\n'.format(best_acc, best_epoch), 'green')
        saver.write('{}\n'.format(val_acc))

    # -- testing
    cprint('>> testing <<', 'cyan')
    test_total = 0
    test_correct = 0

    net.load_state_dict(best_weights)
    net.eval()
    with torch.no_grad():
        for _, (images, labels, _) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            outputs, _ = net(images)

            test_total += images.size(0)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()

    test_acc = test_correct / test_total * 100.

    cprint('>> test accuracy: {}'.format(test_acc), 'cyan')
    saver.write('>> test accuracy: {}\n'.format(test_acc))

    # retest on the validation set, for sanity check
    cprint('>> validation <<', 'cyan')
    val_total = 0
    val_correct = 0
    net.eval()
    with torch.no_grad():
        for _, (images, labels, _) in enumerate(valloader):
            images, labels = images.to(device), labels.to(device)

            outputs, _ = net(images)

            val_total += images.size(0)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = val_correct / val_total * 100.
    cprint('>> validation accuracy: {}'.format(val_acc), 'cyan')
    saver.write('>> validation accuracy: {}'.format(val_acc))

    saver.close()

    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0', help='delimited list input of GPUs', type=str)
    parser.add_argument('--every', default=5, help='collect the big connected component every n epochs', type=int)
    parser.add_argument('--nworker', default=1, help='number of worker', type=int)
    parser.add_argument('--start_clean', default=30, help='when to start collecting clean data', type=int)
    parser.add_argument('--k_outlier', default=32, help='the k for knn in cleaning big connected component', type=int)
    parser.add_argument('--k_cc', default=4, help='the k for knn in collecting big connected component', type=int)
    parser.add_argument('--noise', default=0.4, help='the noise level', type=float)
    parser.add_argument('--type', default='uniform', help='which type of noise [uniform | asym]', type=str)
    parser.add_argument('--patience', default=65, help='if no improvement for consecutive $patience$ epochs, then stop', type=int)
    parser.add_argument('--seed', default=80, help='random seed', type=int)
    parser.add_argument('--dataset', default='cifar10', help='dataset [cifar10 | cifar100 | pc]', type=str)
    parser.add_argument('--milestone', default='60,120', help='milestone', type=str)
    parser.add_argument('--net', default='resnet18', help='network type [resnet18 | resnet34 | pc]', type=str)
    parser.add_argument('--zeta', default=0.5, help='zeta', type=float)

    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    main(args)
