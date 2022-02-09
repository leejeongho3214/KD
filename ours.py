import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import torch.nn.functional as F
from cutout import Cutout
from models.resnet import *
from models.preactresnet import *
from models.senet import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Task-Oriented Feature Distillation. ')
parser.add_argument('--model', default="resnet50", help="choose the student model", type=str)
parser.add_argument('--dataset', default="cifar100", type=str, help="cifar10/cifar100")
parser.add_argument('--alpha', default=0.05, type=float)
parser.add_argument('--beta', default=0.03, type=float)
parser.add_argument('--l2', default=7e-3, type=float)
parser.add_argument('--teacher', default="resnet152", type=str)
parser.add_argument('--t', default=3.0, type=float, help="temperature for logit distillation ")
args = parser.parse_args()

BATCH_SIZE = 64
LR = 0.1
transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),
                                      transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                      Cutout(n_holes=1, length=16),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset, testset = None, None
if args.dataset == 'cifar100':
    trainset = torchvision.datasets.CIFAR100(
        root='./.data',
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root='./.data',
        train=False,
        download=True,
        transform=transform_test
    )
if args.dataset == 'cifar10':
    trainset = torchvision.datasets.CIFAR10(
        root='data',
        train=True,
        download=True,
        transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=transform_test
    )
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

#   get the student model
if args.model == "resnet18":
    student = resnet18()
if args.model == "resnet50":
    student = resnet50()
if args.model == "resnet101":
    student = resnet101()
if args.model == "resnet152":
    student = resnet152()
if args.model == "preactresnet18":
    student = preactresnet18()
    LR = 0.02
    # reduce init lr for stable training
if args.model == "preactresnet50":
    student = preactresnet50()
    LR = 0.02
    # reduce init lr for stable training
if args.model == "senet18":
    student = seresnet18()
if args.model == "senet50":
    student = seresnet50()

#   get the teacher model
if args.teacher == 'resnet18':
    teacher = resnet18()
elif args.teacher == 'resnet50':
    teacher = resnet50()
elif args.teacher == 'resnet101':
    teacher = resnet101()
elif args.teacher == 'resnet152':
    teacher = resnet152()

ss = torch.load("resnet152.pth.tar")
torch.save(ss.state_dict(), "resnet152.pth")
teacher.load_state_dict(torch.load("resnet152.pth"))
teacher.cuda()
student.cuda()
assistant = resnet50()
assistant.cuda()
orthogonal_penalty = args.beta
init = False
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(student.parameters(), lr=LR, weight_decay=args.l2, momentum=0.9)
optimizer2 = optim.SGD(assistant.parameters(), lr=LR, weight_decay=args.l2, momentum=0.9)

if __name__ == "__main__":
    best_acc = 0
    count = 0
    print("Start Training")
    for epoch in range(250):
        if epoch in [80, 160, 240]:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 10
        student.train()
        assistant.train()
        sum_loss = 0.0
        sum_loss2 = 0.0
        correct = 0.0
        correct2= 0.0
        total = 0.0
        endtime = time.time()
        for i, data in enumerate(trainloader, 0):
            torch.cuda.empty_cache()
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            student_outputs, student_feature = student(inputs)
            assistant_outputs, assistant_feature = assistant(inputs)

            #   get teacher results
            with torch.no_grad():
                teacher_logits, teacher_feature = teacher(inputs)

            #   init the feature resizing layer depending on the feature size of students and teachers
            #   a fully connected layer is used as feature resizing layer here
            if not init:
                teacher_feature_size = teacher_feature[0].size(1)
                student_feature_size = student_feature[0].size(1)
                assistant_feature_size = assistant_feature[0].size(1)
                num_auxiliary_classifier = len(teacher_logits)
                link = []
                for j in range(num_auxiliary_classifier):
                    link.append(nn.Linear(student_feature_size, teacher_feature_size, bias=False))
                student.link = nn.ModuleList(link).cuda()

                link2 = []
                for j in range(num_auxiliary_classifier):
                    link2.append(nn.Linear(assistant_feature_size, student_feature_size, bias=False))
                assistant.link = nn.ModuleList(link2).cuda()

                #   we redefine optimizer here so it can optimize the net.link layers.
                optimizer = optim.SGD(student.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)
                optimizer2 = optim.SGD(assistant.parameters(), lr=LR, weight_decay=5e-4, momentum=0.9)

                init = True

            #   compute loss
            loss = torch.FloatTensor([0.]).to(device)

            #   Distillation Loss + Task Loss
            for index in range(len(student_feature)):
                student_feature[index] = student.link[index](student_feature[index])
                #   task-oriented feature distillation loss
                loss += torch.dist(student_feature[index], teacher_feature[index], p=2) * args.alpha
                #   task loss (cross entropy loss for the classification task)
                loss += criterion(student_outputs[index], labels)
                #   logit distillation loss, CrossEntropy implemented in utils.py.
                loss += CrossEntropy(student_outputs[index], teacher_logits[index],
                                     1 + (args.t / 250) * float(1 + epoch))

            # Orthogonal Loss
            for index in range(len(student_feature)):
                weight = list(student.link[index].parameters())[0]
                weight_trans = weight.permute(1, 0)
                ones = torch.eye(weight.size(0)).cuda()
                ones2 = torch.eye(weight.size(1)).cuda()
                loss += torch.dist(torch.mm(weight, weight_trans), ones, p=2) * args.beta
                loss += torch.dist(torch.mm(weight_trans, weight), ones2, p=2) * args.beta



            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # optimize student model
            with torch.no_grad():
                student_outputs2, student_feature2 = student(inputs)

            loss2 = torch.FloatTensor([0.]).to(device)
            for index in range(len(assistant_feature)):
                assistant_feature[index] = assistant.link[index](assistant_feature[index])
                #   task-oriented feature distillation loss
                loss2 += torch.dist(student_feature2[index]-assistant_feature[index],teacher_feature[index] , p=2) * args.alpha
                #   task loss (cross entropy loss for the classification task)
                loss2 += criterion(assistant_outputs[index], labels)
                #   logit distillation loss, CrossEntropy implemented in utils.py.
                loss2 += CrossEntropy(assistant_outputs[index], student_outputs2[index], 1 + (args.t / 250) * float(1 + epoch))


            for index in range(len(assistant_feature)):
                weight = list(assistant.link[index].parameters())[0]
                weight_trans = weight.permute(1, 0)
                ones = torch.eye(weight.size(0)).cuda()
                ones2 = torch.eye(weight.size(1)).cuda()
                loss2 += torch.dist(torch.mm(weight, weight_trans), ones, p=2) * args.beta
                loss2 += torch.dist(torch.mm(weight_trans, weight), ones2, p=2) * args.beta


            optimizer2 = optim.SGD(assistant.parameters(), lr=LR, weight_decay=args.l2, momentum=0.9)

            sum_loss2 += loss2.item()
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            total += float(labels.size(0))
            _, predicted = torch.max(assistant_outputs[0].data, 1)
            correct += float(predicted.eq(labels.data).cpu().sum())

            _, predicted2 = torch.max(student_outputs[0].data, 1)
            correct2 += float(predicted2.eq(labels.data).cpu().sum())

            if i % 20 == 0:
                print('\r[epoch:%d, iter:%d] Loss: %.03f | Assistance Acc: %.2f%% || Student Acc: %.2f%% '
                      % (epoch + 1, (i + 1 + epoch * length), sum_loss2 / (i + 1),
                         100 * correct / total, 100 * correct2 / total), end='')


        check_time = (time.time() - endtime) * (250 - epoch)
        top = time.strftime('%c', time.localtime(time.time() + check_time))
        if epoch == 0:
            print('\rTotal training time is %dh  %dm   %ds' % (
            check_time / 3600, (check_time % 3600) / 60, (check_time % 60)))
            print("\rExpected finished time is %s" % top)
        print("\rRemaining time is %dh  %dm   %ds" % (check_time / 3600, (check_time % 3600) / 60, (check_time % 60)))

        with torch.no_grad():
            correct = 0.0
            correct2 = 0
            total = 0.0

            for data in testloader:
                assistant.eval()
                student.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                assistant_outputs, feature = assistant(images)
                _, predicted = torch.max(assistant_outputs[0].data, 1)
                correct += float(predicted.eq(labels.data).cpu().sum())
                total += float(labels.size(0))

                student_outputs, feature2 = student(images)
                _, predicted2 = torch.max(student_outputs[0].data, 1)
                correct2 += float(predicted2.eq(labels.data).cpu().sum())


            if correct / total < best_acc:

                count += 1
                print("\rDon't upgrade accuracy (%d/60)  assistant acc:  %.4f%%  ||  student acc: %.4f%%" % (count,  (100 * correct/total), correct2* 100 / total))
                if count == 60:
                    break
            else:
                count = 0
                best_acc = correct / total

                print("\r%d Epoch assistant acc:  %.4f%%  || student acc: %.4f%% " % (epoch + 1, (100 * best_acc), (100 * correct2/total)))
                torch.save(assistant.state_dict(), "./checkpoint/assitant_" + args.model + ".pth")

    # print("Training Finished, Best Accuracy is %.4f%%" % (best_acc * 100))
