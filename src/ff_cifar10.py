import numpy as np
import torch

from src import utils

import matplotlib.pyplot as plt


class FF_CIFAR10(torch.utils.data.Dataset):
    def __init__(self, opt, partition, num_classes=10):
        self.opt = opt
        self.cifar10 = utils.get_CIFAR10_partition(opt, partition)
        self.num_classes = num_classes
        self.uniform_label = torch.ones(self.num_classes) / self.num_classes

        # unsupervised helper variables
        self.filter = torch.tensor([[0.0625,0.125,0.0625],[0.125,0.25,0.125],[0.0625,0.125,0.0625]])

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, prediction_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "prediction_sample": prediction_sample,
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.cifar10)

    def _get_pos_sample(self, sample, class_label):
        if self.opt.input.supervised:
            one_hot_label = torch.nn.functional.one_hot(
                torch.tensor(class_label), num_classes=self.num_classes
            )
            pos_sample = sample.clone()
            pos_sample[:, 0, : self.num_classes] = one_hot_label
        else:
            pos_sample = sample.clone()
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        if self.opt.input.supervised:
            # Create randomly sampled one-hot label.
            classes = list(range(self.num_classes))
            classes.remove(class_label)  # Remove true label from possible choices.
            wrong_class_label = np.random.choice(classes)
            one_hot_label = torch.nn.functional.one_hot(
                torch.tensor(wrong_class_label), num_classes=self.num_classes
            )

            neg_sample = sample.clone()
            neg_sample[:, 0, : self.num_classes] = one_hot_label
        else:
            # construct a negative example for unsupervised case (3.2 in paper)
            mask = (torch.rand(self.opt.input.input_width, self.opt.input.input_height) > 0.5).long() # initiate as random bit image

            for i in range(20): # repeat blurring 20 times
                mask = self.blur(mask)

            mask = (mask > 0.5).long() # mask
            mask = mask.unsqueeze(0)

            index = torch.randint(len(self.cifar10), (1,))
            while class_label == self.cifar10[index][1]:
                index = torch.randint(len(self.cifar10),(1,))

            neg_sample = (mask * sample.clone() + (1-mask) * self.cifar10[index][0].clone())

        return neg_sample

    def _get_neutral_sample(self, sample):
        if self.opt.input.supervised:
            neutral_sample = sample.clone()
            neutral_sample[:, 0, : self.num_classes] = self.uniform_label
        else:
            neutral_sample = sample.clone()
        return neutral_sample
    
    def _get_prediction_sample(self, sample):
        return sample.clone()

    def _generate_sample(self, index):
        # Get CIFAR10 sample.
        sample, class_label = self.cifar10[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        prediction_sample = self._get_prediction_sample(sample)
        return pos_sample, neg_sample, neutral_sample, prediction_sample, class_label
    

    

    ## unsupervised helpers
    def blur(self, img):
        # blur img using filter
        m,n = img.shape
        radius = (self.filter.shape[0]-1)//2
        new_img = torch.zeros(m,n)
        for i in range(m):
            for j in range(n):
                top,left,bottom,right = max(0,i-radius),max(0,j-radius),min(m-1,i+radius),min(m-1,j+radius)
                new_img[i][j] = sum([img[x][y]*self.filter[x-i+radius][y-j+radius]
                                    for y in range(left,right+1) for x in range(top,bottom+1)])/ \
                                sum([self.filter[x-i+radius][y-j+radius] for y in range(left,right+1) for x in range(top,bottom+1)])
        return new_img

    def generate_negative_example(self):
        # construct a negative example for unsupervised case (3.2 in paper)
        mask = (torch.rand(28,28) > 0.5).long() # initiate as random bit image

        for i in range(20): # repeat blurring 6 times
            mask = self.blur(mask,filter)

        mask = (mask > 0.5).long() # mask

        index1,index2 = torch.randint(len(self.cifar10),(2,))
        while self.cifar10[index1][1] == self.cifar10[index2][1]:
            index1,index2 = torch.randint(len(self.cifar10),(2,))

        return (mask * self.cifar10[index1][0] + (1-mask) * self.cifar10[index2][0])
