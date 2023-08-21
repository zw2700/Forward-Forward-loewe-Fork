import math
from functools import partial
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD

from src import utils


class FF_model(torch.nn.Module):
    """The model trained with Forward-Forward (FF)."""

    def __init__(self, opt):
        super(FF_model, self).__init__()

        self.opt = opt
        self.num_channels = [self.opt.model.hidden_dim] * self.opt.model.num_blocks
        self.act_fn = ReLU_full_grad()

        # Initialize the model.
        self.model = nn.ModuleList()
        prev_dimension = 784
        for i in range(len(self.num_channels)):
            block = nn.ModuleList([nn.Linear(prev_dimension, self.num_channels[i])])
            for j in range(self.opt.model.num_layers_per_block - 1):
                block.append(nn.Linear(self.num_channels[i], self.num_channels[i]))
            prev_dimension = self.num_channels[i]
            self.model.append(block)

        # Initialize forward-forward loss.
        self.ff_loss = nn.BCEWithLogitsLoss()

        # Initialize peer normalization loss.
        self.running_means = [
            torch.zeros(self.num_channels[i], device=self.opt.device) + 0.5
            for i in range(self.opt.model.num_blocks)
        ]

        # Initialize downstream classification loss.
        channels_for_classification_loss = sum(
            self.num_channels[-i] for i in range(self.opt.model.num_blocks - 1)
        )
        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification_loss, 10, bias=False)
        )
        self.classification_loss = nn.CrossEntropyLoss()

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        for block in self.model.modules():
            for m in block.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.normal_(
                        m.weight, mean=0, std=1 / math.sqrt(m.weight.shape[0])
                    )
                    torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=-1, keepdim=True)) + eps)

    def _calc_peer_normalization_loss(self, idx, z):
        # Only calculate mean activity over positive samples.
        mean_activity = torch.mean(z[:self.opt.input.batch_size], dim=0)

        self.running_means[idx] = self.running_means[
            idx
        ].detach() * self.opt.model.momentum + mean_activity * (
            1 - self.opt.model.momentum
        )

        peer_loss = (torch.mean(self.running_means[idx]) - self.running_means[idx]) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(self, z, labels):
        sum_of_squares = torch.sum(z ** 2, dim=-1)

        logits = sum_of_squares - z.shape[1]
        ff_loss = self.ff_loss(logits, labels.float())

        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels)
                / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy

    def forward(self, inputs, labels):
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "Peer Normalization": torch.zeros(1, device=self.opt.device),
        }

        # Concatenate positive and negative samples and create corresponding labels.
        z = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0)
        posneg_labels = torch.zeros(z.shape[0], device=self.opt.device)
        posneg_labels[: self.opt.input.batch_size] = 1

        xs,us,jvps = [],[],[]

        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        # define partial functions for jvp calculation
        def f(z, block_idx, layer_idx):
            z = self.act_fn.apply(z)
            while layer_idx + 1 < self.opt.model.num_layers_per_block:
                layer_idx += 1
                z = self._layer_norm(z)
                z = self.model[block_idx][layer_idx](z)
                z = self.act_fn.apply(z)
            z = self._calc_ff_loss(z, posneg_labels)[0] \
                    + self.opt.model.peer_normalization * self._calc_peer_normalization_loss(block_idx, z)
            return z

        for block_idx, block in enumerate(self.model):
            block_xs, block_us, block_jvps = [],[],[]

            for layer_idx, layer in enumerate(block[:-1]):
                block_xs.append(torch.mean(z,dim=0))
                z = block[layer_idx](z)
                u = torch.randn(*z.shape)
                block_us.append(torch.mean(u,dim=0))

                # high-level jacobian-vector product
                f_part = partial(f, block_idx=block_idx, layer_idx=layer_idx)
                _, jvp = torch.func.jvp(f_part, (z,), (u,))
                block_jvps.append(jvp)

                z = self.act_fn.apply(z)
                z = z.detach()
                z = self._layer_norm(z)

            # final layer in the block
            z = block[-1](z)
            z = self.act_fn.apply(z)

            if self.opt.model.peer_normalization > 0:
                peer_loss = self._calc_peer_normalization_loss(block_idx, z)
                scalar_outputs["Peer Normalization"] += peer_loss
                scalar_outputs["Loss"] += self.opt.model.peer_normalization * peer_loss

            ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)
            scalar_outputs[f"loss_layer_{block_idx}"] = ff_loss
            scalar_outputs[f"ff_accuracy_layer_{block_idx}"] = ff_accuracy
            scalar_outputs["Loss"] += ff_loss
            
            z = z.detach()
            z = self._layer_norm(z)

            xs.append(block_xs.copy())
            us.append(block_us.copy())
            jvps.append(block_jvps.copy())

        scalar_outputs = self.forward_downstream_classification_model(
            inputs, labels, scalar_outputs=scalar_outputs
        )

        return scalar_outputs, xs, us, jvps

    def forward_downstream_classification_model(
        self, inputs, labels, scalar_outputs=None,
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        z = inputs["neutral_sample"]
        z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        input_classification_model = []

        with torch.no_grad():
            for block_idx, block in enumerate(self.model):
                for layer_idx, layer in enumerate(block):
                    z = layer(z)
                    z = self.act_fn.apply(z)
                    z = self._layer_norm(z)

                if block_idx >= 1:
                    input_classification_model.append(z)

        input_classification_model = torch.concat(input_classification_model, dim=-1)

        output = self.linear_classifier(input_classification_model.detach())
        output = output - torch.max(output, dim=-1, keepdim=True)[0]
        classification_loss = self.classification_loss(output, labels["class_labels"])
        classification_accuracy = utils.get_accuracy(
            self.opt, output.data, labels["class_labels"]
        )

        scalar_outputs["Loss"] += classification_loss
        scalar_outputs["classification_loss"] = classification_loss
        scalar_outputs["classification_accuracy"] = classification_accuracy
        return scalar_outputs


class ReLU_full_grad(torch.autograd.Function):
    """ ReLU activation function that passes through the gradient irrespective of its input value. """

    @staticmethod
    def forward(input):
        return input.clamp(min=0)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.input = inputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()
    
    @staticmethod
    def jvp(ctx, grad_input):
        input = ctx.input
        assert input.shape == grad_input.shape
        grad_in = grad_input.clone()
        grad_in[input<0] = 0
        return grad_in