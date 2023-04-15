import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy
from torchmetrics import F1Score
from torchmetrics.functional import accuracy
from torchmetrics.functional import f1_score


from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


save_model_path = "checkpoints/"
pth_extension = ".pth"


def val(model, data_val, loss_function, writer, epoch):
    accuracyy = 0
    f1_scoree = 0
    f1 = F1Score(num_classes=2, task = 'binary')
    data_iterator = enumerate(data_val)  # take batches
    f1_list = []
    f1t_list = []

    with torch.no_grad():
        model.eval()  # switch model to evaluation mode
        tq = tqdm(total=len(data_val))
        tq.set_description('Validation')

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

            # f1_list.extend(torch.argmax(pred, dim =1).tolist())
            # f1t_list.extend(torch.argmax(label, dim =1).tolist())
            # f1score += f1(label.squeeze().detach().cpu(), pred.squeeze().detach().cpu())
            # Take the argmax of the second dimension to obtain the predicted binary labels
            pred = torch.argmax(pred, dim=1)


            total_loss += loss.item()
            accuracyy += accuracy(pred, label, num_classes=2, task="binary").cuda()
            f1_scoree += f1_score(pred, label, num_classes=2, task="binary").cuda()
            tq.update(1)

    tq.close()
    # print("F1 score: ", f1(torch.tensor(f1_list), torch.tensor(f1t_list)))
    # writer.add_scalar("Validation mIoU", f1(torch.tensor(f1_list), torch.tensor(f1t_list)), epoch)
    #  writer.add_scalar("Validation Loss", total_loss/len(data_val), epoch)
    # f1_score_final = round((f1_scoree.item())/len(data_val),5)
    print("\nF1 score: ", str(round((f1_scoree.item()*100)/len(data_val),2))+'%')
    print("Accuracy: ", str(round((accuracyy.item()*100)/len(data_val),2))+'%\n')
    print("Loss: ", total_loss/len(data_val))

    return None


def train(model, dataloader, val_loader, optimizer, loss_fn, n_epochs, filename):
    device = 'cuda'
    writer = SummaryWriter()

    model.cuda()  # Move the model to the specified device (e.g., GPU or CPU)
    model.train()  # Set the model to training mode
    for epoch in range(n_epochs):
        running_loss = 0.0
        tq = tqdm(total=len(dataloader))
        tq.set_description('epoch %d' % (epoch))
        # optimizer.zero_grad()


        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)  # Move the batch of images to the specified device
            labels = labels.to(device)  # Move the batch of labels to the specified device
            
            optimizer.zero_grad()  # Reset the gradients of the optimizer
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = loss_fn(outputs, labels)
            outputs = outputs.softmax(dim=1)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()
            
            running_loss += loss.item()
            tq.set_postfix(loss_st='%.6f' % loss.item())
            tq.update(1)
            
        tq.close()

        epoch_loss = running_loss / len(dataloader)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, epoch_loss))

        
        f1_score_final = val(model, val_loader, loss_fn, writer, epoch)

        # scheduler.step(f1_score_final)
        print("current learning rate:", optimizer.param_groups[0]['lr'])

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(checkpoint, os.path.join(save_model_path, (filename + pth_extension)))
        print("saved the model " + save_model_path)
        model.train()

if __name__ == '__main__':
    train_data = custom_dataset("train")
    val_data = custom_dataset("val")

    train_loader = DataLoader(
        train_data,
        batch_size=16,
        shuffle=True,
        num_workers = 4
    )

    val_loader = DataLoader(
        val_data,
        batch_size=16,
        num_workers = 4
    )

    
    model_res = ModelRes(2).cuda()
    model_vgg = ModelVGG(2).cuda()
 
    
    # loss function 
    loss = nn.CrossEntropyLoss()

    print('\n----------------------ResNet18, SGD----------------------\n')

    optimizer_sgd = SGD(model_res.parameters(),  lr=0.005)

    train(model_res, train_loader, val_loader, optimizer_sgd, loss, 10, "resnet_sgd")
    

    print('\n----------------------ResNet18, Adam----------------------\n')

    optimizer_adam = Adam(model_res.parameters(), lr=0.005)

    train(model_res, train_loader, val_loader, optimizer_adam, loss, 10, "resnet_adam")
    


    print('\n----------------------VGG16, SGD----------------------\n')

    optimizer_sgd = SGD(model_vgg.parameters(),  lr=0.002)
    # scheduler = ReduceLROnPlateau(optimizer_sgd, mode='max', factor=0.5, patience=1, verbose=True)
    train(model_vgg, train_loader, val_loader, optimizer_sgd, loss, 3, "vgg_sgd")
    

    print('\n----------------------VGG16, Adam----------------------\n')

    optimizer_adam = Adam(model_vgg.parameters(), lr=0.002)
    # scheduler = ReduceLROnPlateau(optimizer_adam, mode='max', factor=0.5, patience=1, verbose=True)
    train(model_vgg, train_loader, val_loader, optimizer_adam, loss, 10, "vgg_adam")



