import datetime
from resnet18 import *
from utils import *
import argparse
import os

parser = argparse.ArgumentParser(description='Project-one')
parser.add_argument('--experiment_name', type=str, required=True)
parser.add_argument('--init_channel_size', type=int, default=16, metavar='N')  # 64 ori done
# parser.add_argument('--activation_function', type=int, default=1, metavar='N')
parser.add_argument('--GAP', type=int, default=0, metavar='N')  # done
parser.add_argument('--batch_size', type=int, default=16, metavar='N')  # 128 ori
parser.add_argument('--learning_rate', type=float, default=0.1, metavar='N')  # 32 ori
parser.add_argument('--epoch', type=int, default=32, metavar='N')
parser.add_argument('--weight_decay', type=float, default=0.0005, metavar='N')
parser.add_argument('--shortcut', type=int, default=1, metavar='N')  # done
parser.add_argument('--batch_normalization', type=int, default=1, metavar='N')  # done
parser.add_argument('--loss_function', type=int, default=0, metavar='N')
# 0 CrossEntropyLoss 1 MSELoss 2 L1Loss
parser.add_argument('--optimizer', type=int, default=1, metavar='N')  # done
# True1 SGD False0  Adam
parser.add_argument('--scheduler', type=int, default=1, metavar='N')
# 0 no scheduler 1 ReduceLROnPlateau 2 StepLR 3 ExponentialLR 4 CosineAnnealingLR
# best result ReduceLROnPlateau
args = parser.parse_args()

setup_seed(6666)

train_loader = load_data(batch_size=args.batch_size)
iter_val_loader = load_data(train=False, batch_size=args.batch_size, n_items=512)
epoch_val_loader, test_loader = load_data(train=False, batch_size=args.batch_size)

if args.shortcut == False:
    net = PlainNet().to(device)
elif args.GAP == True:
    net = ResNet18GAP().to(device)
else:
    net = ResNet18(if_bn=args.batch_normalization, channel=args.init_channel_size).to(device)

if args.loss_function == 0:
    criterion = torch.nn.CrossEntropyLoss()
elif args.loss_function == 1:
    criterion = torch.nn.MSELoss()
else:
    criterion = torch.nn.L1Loss()

if args.optimizer:
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
else:
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, amsgrad=False)

# https://zhuanlan.zhihu.com/p/69411064
if args.scheduler == 0:
    pass
elif args.scheduler == 1:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, min_lr=1e-4)
elif args.scheduler == 2:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
elif args.scheduler == 3:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
else:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0)

loss_list = []
grad_list = []
grad_list2 = []
train_err = []
val_err = []

start_time = datetime.datetime.now()
for epoch in range(args.epoch):
    acc = train_model(epoch, (train_loader, iter_val_loader, epoch_val_loader),
                      (loss_list, grad_list, grad_list2, train_err, val_err),
                      (net, criterion, optimizer))
    if args.scheduler == 0:
        pass
    else:
        scheduler.step(acc)

end_time = datetime.datetime.now()
print('Training time:%d' % (end_time - start_time).seconds)

os.mkdir(os.path.join('experiments', args.experiment_name))
draw_loss(loss_list, args.experiment_name)
draw_acc(train_err, val_err, args.experiment_name)

loss_list = [i.item() for i in loss_list]
ILV, EFR = get_ILV_and_EFR(loss_list, train_err, val_err)
print(f'ILV: {ILV}')
print(f'EFR: {EFR}')
print(f'Accuracy: {acc}')

torch.save(loss_list, os.path.join('experiments', args.experiment_name, 'loss_list'))
torch.save(train_err, os.path.join('experiments', args.experiment_name, 'train_err'))
torch.save(val_err, os.path.join('experiments', args.experiment_name, 'val_err'))
torch.save(grad_list, os.path.join('experiments', args.experiment_name, 'grad_list'))
torch.save(grad_list2, os.path.join('experiments', args.experiment_name, 'grad_list2'))
torch.save(net.state_dict(), os.path.join('experiments', args.experiment_name, 'param.pth'))
