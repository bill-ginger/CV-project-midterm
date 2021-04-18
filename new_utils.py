import torch
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 确保模型可复现
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PartialDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, n_items=10):
        self.dataset = dataset
        self.n_items = n_items

    def __getitem__(self, num):
        return self.dataset.__getitem__(num)

    def __len__(self):
        return min(self.n_items, len(self.dataset))

    def partial(self):
        part_dataset = []
        for i in range(self.__len__()):
            part_dataset.append(self.__getitem__(i))
        return part_dataset


def load_data(data_root="./data", batch_size=128, train=True, n_items=-1):
    normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalization
        ])

    dataset = torchvision.datasets.CIFAR10(root=data_root, train=train, download=False, transform=transform)
    if n_items > 0:
        dataset = PartialDataset(dataset, n_items).partial()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=16)
    return loader




def new_load_data(data_root="./data", batch_size=128, val_size=8192):
    normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalization
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        normalization
    ])
    
    train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, transform=transform_train)
    train_set, val_set = torch.utils.data.random_split(train_set, [50000 - val_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, num_workers=16)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=16)

    return train_loader, val_loader, test_loader










def get_val_acc(model, loader):
    # evaluate the model
    list_loader = [(x_.to(device), l_.to(device)) for (x_, l_) in loader] if not isinstance(loader, list) else loader
    model.eval()
    total_correct = 0
    total_num = 0
    with torch.no_grad():
        for x, label in list_loader:
            # x, label = x.to(device), label.to(device)
            pred = model(x).argmax(dim=1)
            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)
        acc = total_correct / total_num
    return acc


def train_model(epoch, loader, save_lists, model):
    # unpacking parameters
    train_loader, iter_val_loader, epoch_val_loader = loader
    all_loss, train_err, val_err = save_lists
    net, criterion, optimizer = model
    iter_val_list = [(x.to(device), l.to(device)) for (x, l) in iter_val_loader]

    for batch_idx, (x, label) in enumerate(train_loader):
        net.train()
        # forward pass
        x, label = x.to(device), label.to(device)
        pred = net(x)
        loss = criterion(pred, label)
        all_loss.append(loss)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get training accuracy per batch
        correct = torch.eq(pred.argmax(dim=1), label).float().sum().item()
        train_acc = correct / x.shape[0]
        train_err.append(1 - train_acc)

        # get test accuracy per batch
        val_acc = get_val_acc(net, iter_val_list)
        val_err.append(1 - val_acc)

        if batch_idx % 50 == 0:
            print(f'{epoch}\t{batch_idx}\tloss: {loss.item()}\tlr: {optimizer.param_groups[0]["lr"]:.5f}')
            print(f'train acc: {train_acc:.5f}\t validation acc: {val_acc:.5f}')
    # use the validation accuracy for the scheduler to decay learning rate
    epoch_val_acc = get_val_acc(net, epoch_val_loader)
    print(f'epoch:{epoch:4}\tepoch acc: {epoch_val_acc}')
    return epoch_val_acc


def get_ILV_and_EFR(loss_list, train_error_list, val_error_list, stride=1, q=0.9, section_len=16):
    loss_LV = []
    train_acc_LV = []
    val_acc_LV = []

    loss_FR = []
    train_acc_FR = []
    val_acc_FR = []
    x_list = np.linspace(1, section_len, section_len).reshape((-1, 1))

    for index in range((len(loss_list) - section_len) // stride + 1):
        loss_section = loss_list[index:(index + section_len)]
        train_section = train_error_list[index:(index + section_len)]
        val_section = val_error_list[index:(index + section_len)]

        regression = LinearRegression().fit(x_list, loss_section)
        loss_LV.append(-regression.coef_[0] / np.mean(loss_section))
        loss_FR.append(np.sum(np.maximum(np.diff(loss_section, n=1), 0)) / section_len / np.mean(loss_section))

        regression = LinearRegression().fit(x_list, train_section)
        train_acc_LV.append(-regression.coef_[0] / np.mean(train_section))
        train_acc_FR.append(np.sum(np.maximum(np.diff(train_section, n=1), 0)) / section_len / np.mean(train_section))

        regression = LinearRegression().fit(x_list, val_section)
        val_acc_LV.append(-regression.coef_[0] / np.mean(val_section))
        val_acc_FR.append(np.sum(np.maximum(np.diff(val_section, n=1), 0)) / section_len / np.mean(val_section))

    q_list = np.logspace(1, len(loss_LV), num=len(loss_LV), base=q)
    q_list = q_list / np.sum(q_list)

    loss_ILV = np.sum(loss_LV * q_list)
    train_accuracy_ILV = np.sum(train_acc_LV * q_list)
    val_accuracy_ILV = np.sum(val_acc_LV * q_list)

    # q_list2 = np.logspace(len(loss_LV), 1, num=len(loss_LV), base=q)
    # q_list2 = q_list2 / np.sum(q_list2)

    loss_EFR = np.sum(loss_FR * q_list)
    train_accuracy_EFR = np.sum(train_acc_FR * q_list)
    val_accuracy_EFR = np.sum(val_acc_FR * q_list)

    ILV = {"loss": loss_ILV, "train": train_accuracy_ILV, "val": val_accuracy_ILV}
    EFR = {"loss": loss_EFR, "train": train_accuracy_EFR, "val": val_accuracy_EFR}
    return ILV, EFR


def draw_loss(loss_list, filename='Loss.jpg', title='Loss', save=True):
    plt.figure(dpi=196)
    plt.plot(range(len(loss_list)), loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    if save:
        plt.savefig(filename)
    plt.show()


def draw_acc(train_err, val_err, filename='Accuracy.jpg', save=True):
    train_acc = [1-err for err in train_err]
    val_acc = [1-err for err in val_err]
    plt.figure(dpi=196)
    plt.plot(range(len(train_acc)), train_acc, c='g', label='Training')
    plt.plot(range(len(val_acc)), val_acc, c='b', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()
    if save:
        plt.savefig(filename)
    plt.show()
