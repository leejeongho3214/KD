import argparse
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from cutout import Cutout
from models.resnet import *
from models.resnet import Bottleneck
from utils import CrossEntropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 200
BATCH_SIZE = 128
LR = 0.1

parser = argparse.ArgumentParser(description='Task-Oriented Feature Distillation. ')
parser.add_argument('--model', default="resnet50", help="choose the student model", type=str)
parser.add_argument('--dataset', default="cifar100", type=str, help="cifar10/cifar100")
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--beta', default=0.03, type=float)
parser.add_argument('--l2', default=7e-3, type=float)
parser.add_argument('--teacher', default="resnet152", type=str)
parser.add_argument('--t', default=3.0, type=float, help="temperature for logit distillation ")
args = parser.parse_args()

transform_train = transforms.Compose(
    [transforms.RandomCrop(32, padding=4, fill=128), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
     Cutout(n_holes=1, length=16), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

trainset, testset = None, None
if args.dataset == 'cifar100':
    trainset = datasets.CIFAR100(root='./.data', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR100(root='./.data', train=False, download=True, transform=transform_test)
if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

if args.model == "resnet18":
    net = resnet18()
if args.model == "resnet50":
    net = resnet50()
if args.model == "resnet101":
    net = resnet101()
if args.model == "resnet152":
    net = resnet152()

if args.teacher == 'resnet18':
    teacher = resnet18()
elif args.teacher == 'resnet50':
    teacher = resnet50()
elif args.teacher == 'resnet101':
    teacher = resnet101()
elif args.teacher == 'resnet152':
    teacher = resnet152()

teacher.load_state_dict(torch.load("./resnet152.pth"))
teacher.cuda()
net.to(device)
orthogonal_penalty = args.beta
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=args.l2, momentum=0.9)


def train(net, trainloader, optimizer, epoch):
    if epoch in [80, 160, 240]:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10
    init = False
    net.train()
    sum_loss = 0.0
    correct = 0.0
    total = 0.0
    for i, data in enumerate(trainloader):
        length = len(trainloader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs, student_feature = net(inputs)

        with torch.no_grad():
            teacher_logits, teacher_feature = teacher(inputs)

        if not init:
            teacher_feature_size = teacher_feature[0].size(1)
            student_feature_size = student_feature[0].size(1)
            num_auxiliary_classifier = len(teacher_logits)
            link = []
            for j in range(num_auxiliary_classifier):
                link.append(nn.Linear(student_feature_size, teacher_feature_size, bias=False))
            net.link = nn.ModuleList(link)
            net.cuda()
            #   we redefine optimizer here so it can optimize the net.link layers.
            optimizer = optim.SGD(net.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)


        loss = torch.FloatTensor([0.]).to(device)

        for index in range(len(student_feature)):
            student_feature[index] = net.link[index](student_feature[index])
            #   task-oriented feature distillation loss
            loss += torch.dist(student_feature[index], teacher_feature[index], p=2) * args.alpha
            #   task loss (cross entropy loss for the classification task)
            loss += criterion(outputs[index], labels)
            #   logit distillation loss, CrossEntropy implemented in utils.py.
            loss += CrossEntropy(outputs[index], teacher_logits[index], 1 + (args.t / 250) * float(1 + epoch))

        for index in range(len(student_feature)):
            weight = list(net.link[index].parameters())[0]
            weight_trans = weight.permute(1, 0)
            ones = torch.eye(weight.size(0)).cuda()
            ones2 = torch.eye(weight.size(1)).cuda()
            loss += torch.dist(torch.mm(weight, weight_trans), ones, p=2) * args.beta
            loss += torch.dist(torch.mm(weight_trans, weight), ones2, p=2) * args.beta

        sum_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = torch.max(outputs[0].data, 1)
        correct += float(predicted.eq(labels.data).cpu().sum())

        print(f'\r[Train] epoch[{epoch + 1}] - '
              f'iteration = {i + 1 + epoch * length}  '
              f'loss = {sum_loss / (i + 1)}  '
              f'acc = {100 * correct / total}% ',end='')

def test(net, testloader):
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for data in testloader:
            net.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs, feature = net(images)
            _, predicted = torch.max(outputs[0].data, 1)
            correct += float(predicted.eq(labels.data).cpu().sum())
            total += float(labels.size(0))
            print('\rTest Set AccuracyAcc:  %.4f%% ' % (100 * float(predicted.eq(labels.data).cpu().sum()) / float(labels.size(0))),end='')

        print('\r[TEST]Test Set AccuracyAcc:  %.4f%% ' % (100 * correct / total))
        test_acc = 100 * correct / total



        return test_acc

def main():
    init = False
    best_acc = 0
    print("start Training")
    for epoch in range(250):
        train(net, trainloader, optimizer, epoch)

        test_acc = test(net,testloader)

        if test_acc > best_acc:
            best_acc = test_acc
            print("Best Accuracy Updated: ", best_acc)
            torch.save(net.state_dict(), "./checkpoint/" + args.model + ".pth")

    print("Training Finished, Best Accuracy is %.4f%%" % (best_acc))


if __name__ == '__main__':
    main()
