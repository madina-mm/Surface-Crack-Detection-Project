import torch 
import torch.nn
from PIL import Image
from torchvision import transforms
import os

from torch.utils.data import Dataset
import random
import numpy as np


class custom_dataset(Dataset):

    # initialize your dataset class
    def __init__(self, mode ="train", image_path = "datasets/surface", label_path = "datasets/surface"):
        self.mode = mode    # you have to specify which set do you use, train, val or test
        self.image_path = image_path  
        self.label_path = label_path

        #create list of paths for the images: 

        self.image_list = []
        self.labels = []

        
        self.unique_labels = []


        for i in os.listdir(image_path):  # i has 2 values : Negative, Poitive
            self.unique_labels.append(i)
            for j in os.listdir(image_path + "/" + i): # j gets name of the each image in i folder.
                filepath = self.image_path + "/" + i + "/" + j
                if not filepath.endswith('.db'):
                    try:
                        image = Image.open(filepath)
                        self.labels.append(i)  # each label is appended to the list - ['Negative', 'Negative', ...., 'Poitive', 'Poitive']
                        self.image_list.append(filepath) # image list contains names of images : example - datasets/surface/Negative/1
                    except:
                        print('there s problem in photo itself', filepath)
                        os.remove(filepath)
                else:
                    print('it is not jpg file', filepath)
                    os.remove(filepath)
        
        print('All images are read')
        # print class balance
        # for i in os.listdir(image_path):
        #     print(i, self.labels.count(i))


        # distribute to val train and test

        # train_size = 22000
        train_size = 28000
        val_size = 6000
        test_size = 6000

        random.seed(42)

        # Combine the image paths and labels into tuples
        data = list(zip(self.image_list, self.labels))

        # Shuffle the data randomly
        random.shuffle(data)

        # Split the dataset into training (22,000), testing (2,000), and validation (1,000) sets
        train_data = data[:train_size]  # [0:22000]
        test_data = data[train_size:train_size+test_size] # [22000:24000]
        val_data = data[-val_size:] # [24000:] or [-1000:]

        # Extract the image paths and labels from the resulting sets
        train_images, train_labels = zip(*train_data)
        test_images, test_labels = zip(*test_data)
        val_images, val_labels = zip(*val_data)


        if(mode == "train"):
            self.image_list = train_images
            self.labels = train_labels
        elif(mode == "val"):
            self.image_list = val_images
            self.labels = val_labels
        else:
            self.image_list = test_images
            self.labels = test_labels

        print('Train, test, val lists are ready')

    def __getitem__(self, index):

        # getitem is required field for pytorch dataloader. 

        image = Image.open(self.image_list[index])
        # image.show()
        image = image.convert("RGB")
        label = self.labels[index]
        label = self.parse_labels(label)
        label = torch.as_tensor(label)
        
        
        # all labels should be converted from any data type to tensor
        # for parallel processing
        if self.mode == "train":
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.494, 0.456, 0.406], std=[0.206, 0.206, 0.206])
            ]) 
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.494, 0.456, 0.406], std=[0.206, 0.206, 0.206])
            ])
                            
        image = transform(image) 

        return image, label  

    def parse_labels(self, label):
        for i in range(len(self.unique_labels)):
            if label == self.unique_labels[i]:
                return i
                
    # def parse_labels(self, label):
    #     if label == "Cat":
    #         return 0
    #     elif label == "Dog":
    #         return 1
    #     else:
    #         raise ValueError("Invalid label: {}".format(label))

    # def parse_labels(self, label):
    #     arr = np.zeros((len(self.unique_labels),), dtype= float)
    #     for i in range(len(self.unique_labels)):
    #         if label == self.unique_labels[i]:
    #             arr[i] = 1.0
    #     return arr

    
    def __len__(self):
        return len(self.image_list)



# custom_dataset()[1]

