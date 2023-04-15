import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy
from torchmetrics import F1Score
from torchmetrics.functional import accuracy
from torchmetrics.functional import f1_score


from torch.utils.data import DataLoader, Dataset
from models.model_resnet import ModelRes
from models.model_vgg import ModelVGG
from datasets.dataset_retrieval import custom_dataset
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import os

import warnings

# Filter warnings
warnings.filterwarnings("ignore")

def test(model, data_test, loss_function, writer, epoch):
    accuracyy = 0
    f1_scoree = 0
    f1 = F1Score(num_classes=2, task = 'binary')
    data_iterator = enumerate(data_test)  # take batches
    f1_list = []
    f1t_list = []

    with torch.no_grad():
        model.eval()  # switch model to evaluation mode
        tq = tqdm(total=len(data_test))
        tq.set_description('Testing')

        total_loss = 0

        for _, batch in data_iterator:
            # forward propagation
            image, label = batch
            image = image.cuda()
            label = label.cuda()
            pred = model(image)

            loss = loss_function(pred, label)
            loss = loss.cuda()

            pred = pred.softmax(dim=1)
            #confidence 

            pred = torch.argmax(pred, dim=1)

            # label_new = torch.tensor([label.item(), label.item()])

            
            # f1_list.extend(torch.argmax(pred, dim =1).tolist())
            # f1t_list.extend(torch.argmax(torch.unsqueeze(label_new, dim=0), dim=1).tolist())

            total_loss += loss.item()
            accuracyy += accuracy(pred, label, num_classes=2, task="binary").cuda()
            f1_scoree += f1_score(pred, label, num_classes=2, task="binary").cuda()
            
            tq.update(1)

    tq.close()
    # print("F1 score: ", f1(torch.tensor(f1_list), torch.tensor(f1t_list)))
    # writer.add_scalar("Validation mIoU", f1(torch.tensor(f1_list), torch.tensor(f1t_list)), epoch)
    # writer.add_scalar("Validation Loss", total_loss/len(data_test), epoch)
    print("\nF1 score: ", str(round((f1_scoree.item()*100)/len(data_test),2))+'%')
    print("Accuracy: ", str(round((accuracyy.item()*100)/len(data_test),2))+'%\n')
    print("Loss: ", total_loss/len(data_test))

    return None

if __name__ == '__main__':
    test_data = custom_dataset("test")

    test_loader = DataLoader(
        test_data,
        batch_size=1,
        shuffle=True,
        num_workers = 4
    )

    model_res = ModelRes(2).cuda()
    model_vgg = ModelVGG(2).cuda()
    #optimizer_sgd = SGD(model.parameters(),  lr=0.001)
    optimizer_sgd = SGD(model_res.parameters(), lr=0.0005)
    optimizer_adam = Adam(model_res.parameters(), lr=0.0005)

    # print("\n---------------Resnet, SGD------------")
    # checkpoint = torch.load("checkpoints/resnet_sgd.pth")
    
    # model_res.load_state_dict(checkpoint['state_dict'])
    # optimizer_sgd.load_state_dict(checkpoint['optimizer'])
    
    # loss = nn.CrossEntropyLoss()
    # writer = SummaryWriter()

    # test(model_res, test_loader, loss, writer, 1)

    # print("\n---------------Resnet, Adam------------")
    # checkpoint = torch.load("checkpoints/resnet_adam.pth")
    # model_res.load_state_dict(checkpoint['state_dict'])
    # optimizer_adam.load_state_dict(checkpoint['optimizer'])
    
    # loss = nn.CrossEntropyLoss()
    # writer = SummaryWriter()

    # test(model_res, test_loader, loss, writer, 1)

    print("\n---------------VGG, SGD------------")
    checkpoint = torch.load("checkpoints/vgg_sgd.pth")

    model_vgg.load_state_dict(checkpoint['state_dict'])
    optimizer_sgd.load_state_dict(checkpoint['optimizer'])
    
    loss = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    test(model_vgg, test_loader, loss, writer, 1)

    print("\n---------------VGG, Adam------------")
    checkpoint = torch.load("checkpoints/vgg_adam.pth")
    
    model_vgg.load_state_dict(checkpoint['state_dict'])
    optimizer_adam.load_state_dict(checkpoint['optimizer'])
    
    loss = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    test(model_vgg, test_loader, loss, writer, 1)