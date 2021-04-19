import datetime
from resnet18 import *
from new_utils import *
setup_seed(6666)


train_loader = load_data(data_root="../data")
iter_val_loader = load_data(data_root="../data", train=False, n_items=512)
epoch_val_loader, test_loader = load_data(data_root="../data", train=False)

net = ResNet18().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, min_lr=1e-4)

loss_list = []
train_err = []
val_err = []

start_time = datetime.datetime.now()
for epoch in range(33):
    acc = train_model(epoch, (train_loader, val_loader, epoch_val_loader), (loss_list, train_err, val_err), (net, criterion, optimizer))
    scheduler.step(acc)
end_time = datetime.datetime.now()
print('Training time:%d' % (end_time - start_time).seconds)

draw_loss(loss_list)
draw_acc(train_err, val_err)

loss_list = [i.item() for i in loss_list]
ILV, EFR = get_ILV_and_EFR(loss_list, train_err, val_err)
print(ILV)
print(EFR)

torch.save(loss_list, './loss_list')
torch.save(train_err, './train_err')
torch.save(val_err, './val_err')
torch.save(net.state_dict(), './param.pth')
