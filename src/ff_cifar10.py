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
        pos_sample, neg_sample, neutral_sample, original_sample, class_label = self._generate_sample(
            index
        )

        inputs = {
            "pos_images": pos_sample,
            "neg_images": neg_sample,
            "neutral_sample": neutral_sample,
            "original_sample": original_sample,
        }
        labels = {"class_labels": class_label}
        return inputs, labels

    def __len__(self):
        return len(self.cifar10)

    def _get_pos_sample(self, sample, class_label):
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(class_label), num_classes=self.num_classes
        )
        pos_sample = sample.clone()
        pos_sample[:, 0, : self.num_classes] = one_hot_label
        return pos_sample

    def _get_neg_sample(self, sample, class_label):
        # Create randomly sampled one-hot label.
        classes = list(range(self.num_classes))
        classes.remove(class_label)  # Remove true label from possible choices.
        wrong_class_label = np.random.choice(classes)
        one_hot_label = torch.nn.functional.one_hot(
            torch.tensor(wrong_class_label), num_classes=self.num_classes
        )

        neg_sample = sample.clone()
        neg_sample[:, 0, : self.num_classes] = one_hot_label
        return neg_sample

    def _get_neutral_sample(self, sample):
        neutral_sample = sample.clone()
        neutral_sample[:, 0, : self.num_classes] = self.uniform_label
        return neutral_sample

    def _get_original_sample(self, sample):
        return sample.clone()


    def _generate_sample(self, index):
        # Get CIFAR10 sample.
        sample, class_label = self.cifar10[index]
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        original_sample = self._get_original_sample(sample)
        return pos_sample, neg_sample, neutral_sample, original_sample, class_label
    