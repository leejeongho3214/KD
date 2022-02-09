import torch

import torch.nn as nn

import time

import torch.optim as optim

import torch.nn.functional as F

from torchvision import transforms, datasets, models

from utils import *

from cutout import Cutout

from models.resnet import *


USE_CUDA = torch.cuda.is_available()

DEVICE = torch.device("cuda" if USE_CUDA else "cpu")



EPOCHS = 125

BATCH_SIZE = 64



#학습 데이터셋 준비 및 미니배치 처리를 위한 데이터로더 생성

train_loader = torch.utils.data.DataLoader(

    datasets.CIFAR100('./.data',    #CIFAR-10 데이터셋 로드

        train = True,

        download = True,

        transform = transforms.Compose([

            #학습 효과를 늘려주기 위해 학습할 이미지량 늘려주기

            transforms.RandomCrop(32, padding = 4),    #이미지 무작위로 자르기

            transforms.RandomHorizontalFlip(),    #이미지 뒤집기​

            transforms.ToTensor(),
            Cutout(n_holes=1, length=16),

            #numpy image : H x W x C

            # torch image : C X H X W

            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),

    batch_size = BATCH_SIZE, shuffle = True)



#실험 데이터셋 준비 및 미니배치 처리를 위한 데이터로더 생성

test_loader = torch.utils.data.DataLoader(

    datasets.CIFAR100('data',    #CIFAR-10 데이터셋 로드

        train = False,

        download = True,

        transform = transforms.Compose([

            transforms.ToTensor(),

            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),

    batch_size = BATCH_SIZE, shuffle = True)





# model = resnet152().to(DEVICE)
model = torch.load("./checkpoint/resnet50.pth.tar")
# model = torch.load("./resnet152.pth.tar")
# model = resnet18().to(DEVICE)

# torch.save(model.state_dict(),"./resnet152.pth")

optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 0.0005)

#50번 호출 될 때마다 학습률에 0.1(gamma 값)을 곱해주어서 학습률이 계속 낮아지는 학습률 감소 기법 사용

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 50, gamma = 0.1)



#학습

def train(model, train_loader, optimizer, epoch):

    model.train()

    endtime = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()

        output = model(data)[0][0]

        loss = F.cross_entropy(output, target)

        loss.backward()

        optimizer.step()
    return endtime
#측정

def evaluate(model, test_loader):

    model.eval()

    test_loss = 0

    correct = 0

    # check_time = (time.time() - endtime) * (250 - epoch)
    # top = time.strftime('%c', time.localtime(time.time() + check_time))
    # if epoch == 1:
    #     print('\rTotal traning time is %dh  %dm   %ds' % (check_time / 3600, (check_time % 3600) / 60, (check_time % 60)))
    #     print("\rExpected finished time is %s" % top)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            output = model(data)[0][0]

            # 배치 오차를 합산

            test_loss += F.cross_entropy(output, target, reduction="sum").item()

            # 가장 높은 값을 가진 인덱스가 바로 예측값

            pred = output.max(1, keepdim=True)[1]

            correct += pred.eq(target.view_as(pred)).sum().item()

    #테스트 로스값

    test_loss /= len(test_loader.dataset)

    test_accuracy = 100. * correct / len(test_loader.dataset)

    print("[{}] Test Loss: {:.4f}, Accuracy : {:.2f}%".format(epoch, test_loss, test_accuracy))

    return test_loss, test_accuracy


best = 0
for epoch in range(1, EPOCHS):

    # endtime = train(model, train_loader, optimizer, epoch)
    # scheduler.step()

    # test_loss, test_accuracy = evaluate(model, test_loader)
    # model = torch.load("./resnet50.pth.tar")
    # test_loss, test_accuracy = evaluate(model, test_loader)
    # model = torch.load("./resnet152.pth.tar")
    # test_loss, test_accuracy = evaluate(model, test_loader)
    # model = torch.load("./checkpoint/resnet18.pth.tar")
    # test_loss, test_accuracy = evaluate(model, test_loader)
    model = torch.load("./resnet18.pth.tar")
    test_loss, test_accuracy = evaluate(model, test_loader)
    break

    # if best < test_accuracy:
    #     best = test_accuracy
    #     torch.save(model, './resnet152.pth.tar')


    # print("[{}] Test Loss: {:.4f}, Accuracy : {:.2f}%".format(epoch, test_loss, test_accuracy))
