import math
from functools import partial
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from src import utils


class FF_model(torch.nn.Module):
    """The model trained with Forward-Forward (FF)."""

    def __init__(self, opt):
        super(FF_model, self).__init__()

        self.opt = opt
        self.act_fn = ReLU_full_grad()

        if not self.opt.model.convolutional:
            self.num_channels = [getattr(self.opt.model.fully_connected, f"hidden_dim_{i+1}", 2000)
                                 for i in range(self.opt.model.num_blocks)]

            # Initialize the model.
            self.model = nn.ModuleList()
            self.bn = []
            prev_dimension = opt.input.input_width * opt.input.input_height * opt.input.input_channels
            for i in range(len(self.num_channels)):
                block = nn.ModuleList([nn.Linear(prev_dimension, self.num_channels[i])])
                for j in range(self.opt.model.fully_connected.num_layers_per_block - 1):
                    block.append(nn.Linear(self.num_channels[i], self.num_channels[i]))
                prev_dimension = self.num_channels[i]
                self.model.append(block)
                self.bn.append(nn.BatchNorm1d(self.num_channels[i]))
        else:
            self.num_channels = [getattr(self.opt.model.conv, f"channels_{i+1}", 1) * (getattr(self.opt.model.conv, f"output_size_{i+1}", 1)**2)
                                 for i in range(self.opt.model.num_blocks)]

            self.model = nn.ModuleList()
            self.bn = []
            prev_dimension = self.opt.input.input_channels
            for i in range(self.opt.model.num_blocks):
                # self.model.append(nn.ModuleList([LocallyConnected2d(prev_dimension,  # in_channels
                #                                                     getattr(self.opt.model.conv, f"channels_{i+1}", 1),     # out_channels
                #                                                     getattr(self.opt.model.conv, f"output_size_{i+1}", 1),  # output_size
                #                                                     getattr(self.opt.model.conv, f"kernel_size_{i+1}", 1),  # kernel_size
                #                                                     getattr(self.opt.model.conv, f"stride_{i+1}", 1),       # stride
                #                                                     getattr(self.opt.model.conv, f"padding_{i+1}", 1)       # padding
                #                                                     )])) 
                self.model.append(nn.ModuleList([nn.Conv2d(prev_dimension,                                       # in_channels
                                                        getattr(self.opt.model.conv, f"channels_{i+1}", 1),     # out_channels
                                                        getattr(self.opt.model.conv, f"kernel_size_{i+1}", 1),  # kernel_size
                                                        stride=getattr(self.opt.model.conv, f"stride_{i+1}", 1),       # stride
                                                        padding=getattr(self.opt.model.conv, f"padding_{i+1}", 1)       # padding
                                                        )]))
                self.bn.append(nn.BatchNorm2d(getattr(self.opt.model.conv, f"channels_{i+1}", 1)))
                prev_dimension = getattr(self.opt.model.conv, f"channels_{i+1}", 1)

        if "cuda" in opt.device:
            for batchnorm in self.bn:
                batchnorm = batchnorm.cuda()

        # Initialize decoder and pooling layers for prediction loss.
        self.pred_loss = nn.CrossEntropyLoss()
        self.pred_decoder = nn.Linear(opt.model.pred_decoder_size, opt.input.num_classes)
        if "cuda" in opt.device:
            self.pred_decoder = self.pred_decoder.cuda()
        if self.opt.model.convolutional:
            self.avgpools = []
            for i in range(len(self.num_channels)):
                reduce_factor = self.num_channels[i] // opt.model.pred_decoder_size
                reduce_log2_factor = int(math.log2(reduce_factor))
                if reduce_log2_factor % 2 > 0:
                    avgpool = nn.AvgPool2d(kernel_size=(2**(reduce_log2_factor//2), 2**(reduce_log2_factor//2+1)))
                else:
                    avgpool = nn.AvgPool2d(kernel_size=(2**(reduce_log2_factor//2), 2**(reduce_log2_factor//2)))
                if "cuda" in opt.device:
                    avgpool = avgpool.cuda()
                self.avgpools.append(avgpool)

        # for block in self.model:
        #     for layer in block:
        #         print(layer.weight.shape)

        # Initialize forward-forward loss.
        self.ff_loss = nn.BCEWithLogitsLoss()

        # Initialize peer normalization loss.
        if not self.opt.model.convolutional:
            self.running_means = [
                torch.zeros(self.num_channels[i], device=self.opt.device) + 0.5
                for i in range(self.opt.model.num_blocks)
            ]
        else:
            self.running_means = [
                torch.zeros(getattr(self.opt.model.conv, f"channels_{i+1}", 1), 
                            getattr(self.opt.model.conv, f"output_size_{i+1}", 1), 
                            getattr(self.opt.model.conv, f"output_size_{i+1}", 1), device=self.opt.device) + 0.5
                for i in range(self.opt.model.num_blocks)
            ]

        # Initialize downstream classification loss.
        channels_for_classification_loss = sum(
            self.num_channels[-i] for i in range(1, self.opt.model.num_blocks)
        ) if self.opt.model.num_blocks > 1 else self.num_channels[0]
        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification_loss, opt.input.num_classes, bias=False)
        )
        self.classification_loss = nn.CrossEntropyLoss()

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        for block in self.model:
            for m in block:
                if not self.opt.model.convolutional:
                    if isinstance(m, nn.Linear):
                        if self.opt.training.init == "He":
                            torch.nn.init.normal_(
                                m.weight, mean=0, std=math.sqrt(2) / math.sqrt(m.weight.shape[1])
                            )
                        elif self.opt.training.init == "Xavier":
                            torch.nn.init.normal_(
                                m.weight, mean=0, std=math.sqrt(1) / math.sqrt(m.weight.shape[1])
                            )
                        torch.nn.init.zeros_(m.bias)
                else:
                    if isinstance(m, LocallyConnected2d):
                        if self.opt.training.init == "He":
                            torch.nn.init.normal_(
                                m.weight, mean=0, std=math.sqrt(2) / math.sqrt(m.weight.shape[-1]*m.weight.shape[-4])
                                # m.weight, mean=0, std=math.sqrt(2) / math.sqrt(m.weight.shape[-2]*m.weight.shape[-3]*m.weight.shape[-5])
                            )
                        elif self.opt.training.init == "Xavier":
                            torch.nn.init.normal_(
                                m.weight, mean=0, std=math.sqrt(1) / math.sqrt(m.weight.shape[-1]*m.weight.shape[-4])
                                # m.weight, mean=0, std=math.sqrt(1) / math.sqrt(m.weight.shape[-2]*m.weight.shape[-3]*m.weight.shape[-5])
                            )
                        torch.nn.init.zeros_(m.bias)
                    elif isinstance(m, nn.Conv2d):
                        if self.opt.training.init == "He":
                            # torch.nn.init.normal_(
                            #     m.weight, mean=0, std=math.sqrt(2) / math.sqrt(m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3])
                            # )
                            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                        elif self.opt.training.init == "Xavier":
                            # torch.nn.init.normal_(
                            #     m.weight, mean=0, std=math.sqrt(1) / math.sqrt(m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3])
                            # )
                            torch.nn.init.xavier_normal_(m.weight)
                        torch.nn.init.zeros_(m.bias)

        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _layer_norm(self, z, eps=1e-8):
        return z / (torch.sqrt(torch.mean(z ** 2, dim=list(range(1,len(z.shape))), keepdim=True)) + eps)

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
        sum_of_squares = torch.sum(z ** 2, dim=list(range(1,len(z.shape))))

        logits = sum_of_squares - (z.reshape(len(z),-1).shape[1] * 0.625)
        # logits = sum_of_squares - z.reshape(len(z),-1).shape[1]
        # print(z.shape, torch.mean(sum_of_squares))
        # print(logits)
        ff_loss = self.ff_loss(logits, labels.float())

        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels)
                / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy
    
    def _calc_pred_loss(self, z, labels, layer_index):
        if self.opt.model.convolutional:
            z = self.avgpools[layer_index](z)
            z = z.reshape(z.shape[0], -1)

        z_decoded = self.pred_decoder(z)
        pred_loss = self.pred_loss(z_decoded, labels)

        with torch.no_grad():
            pred_accuracy = (
                torch.sum(torch.argmax(z_decoded, dim=1) == labels)
                / z.shape[0]
            ).item()
        return pred_loss, pred_accuracy

    def forward(self, inputs, labels):
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "Peer Normalization": torch.zeros(1, device=self.opt.device),
        }

        if self.opt.training.sim_pred.beta == 1:  # FF only
            x = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0)  # for FF only
            posneg_labels = torch.zeros(x.shape[0], device=self.opt.device)  # for FF only
            # posneg_labels[: self.opt.input.batch_size] = 1  # for maximizing goodness for positive samples
            posneg_labels[self.opt.input.batch_size:] = 1  # for minimizing goodness for positive samples
        elif self.opt.training.sim_pred.beta == 0:  # prediction only
            x = inputs["prediction_sample"]
        else:  # simpred
            # Concatenate positive and negative samples and create corresponding labels.
            ff_x = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0)
            posneg_labels = torch.zeros(ff_x.shape[0], device=self.opt.device)
            posneg_labels[: self.opt.input.batch_size] = 1

            # Prediction sample
            pred_x = inputs["prediction_sample"]

            # Comcatenate samples for FF and pred
            x = torch.cat([ff_x, pred_x], dim=0)

        print(x.shape)

        xs,us,jvps = [],[],[]

        if not self.opt.model.convolutional:
            x = x.reshape(x.shape[0], -1)
        x = self._layer_norm(x)

        for block_idx, block in enumerate(self.model):

            # block_xs, block_us, block_jvps = [],[],[]

            # two-layer implementation

            # backward for two layers

            z = block[0](x)
            z = self.bn[block_idx](z)
            z = self.act_fn.apply(z)
            if self.opt.training.dropout > 0:
                z = F.dropout(z, p=self.opt.training.dropout, training=True)
            # z = self._layer_norm(z)
            # z = block[1](z)
            # z = self.act_fn.apply(z)
                
            if 0 < self.opt.training.sim_pred.beta < 1:  # simpred, separate FF and pred inputs
                ff_z, pred_z = z[:-self.opt.input.batch_size], z[-self.opt.input.batch_size:]
                
            if self.opt.training.sim_pred.beta < 1:  # prediction loss, pred or simpred
                if self.opt.training.sim_pred.beta > 0:
                    pred_loss, pred_accuracy = self._calc_pred_loss(pred_z, labels["class_labels"], block_idx)
                else:
                    pred_loss, pred_accuracy = self._calc_pred_loss(z, labels["class_labels"], block_idx)  # for prediction only
                scalar_outputs[f"pred_loss_layer_{block_idx}"] = pred_loss
                scalar_outputs[f"pred_accuracy_layer_{block_idx}"] = pred_accuracy
                scalar_outputs["Loss"] += (1-self.opt.training.sim_pred.beta) * pred_loss

            if self.opt.training.sim_pred.beta > 0:  # peer normalization & FF loss, ff or simpred
                # peer normalization loss
                if self.opt.model.peer_normalization > 0:
                    if self.opt.training.sim_pred.beta < 1:
                        peer_loss = self._calc_peer_normalization_loss(block_idx, ff_z)
                    else:
                        peer_loss = self._calc_peer_normalization_loss(block_idx, z)  # for FF only
                    scalar_outputs["Peer Normalization"] += peer_loss
                    scalar_outputs["Loss"] += self.opt.model.peer_normalization * peer_loss

                # FF loss
                if self.opt.training.sim_pred.beta < 1:
                    ff_loss, ff_accuracy = self._calc_ff_loss(ff_z, posneg_labels)
                else:
                    ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels)  # for FF only
                scalar_outputs[f"loss_layer_{block_idx}"] = ff_loss
                scalar_outputs[f"ff_accuracy_layer_{block_idx}"] = ff_accuracy
                scalar_outputs["Loss"] += self.opt.training.sim_pred.beta * ff_loss

            x = z.detach()
            x = self._layer_norm(x)

            if self.opt.model.convolutional and (block_idx+1) in self.opt.model.conv.pool:
                x = F.max_pool2d(x, 2, 2)  # maxpool

            # forward for one layer, backward for one layer, implement later
            # block_xs.append(x)
            # z = block[0](x)
            # u = torch.randn(*z.shape, device=self.opt.device)
            # u[z<0] = 0
            # block_us.append(u)

            # with fwAD.dual_level():
            #     dual_z = fwAD.make_dual(z, u)
            #     # remainder of first layer
            #     dual_z = self.act_fn.apply(dual_z)
            #     dual_z = fwAD.make_dual(fwAD.unpack_dual(dual_z).primal.detach(),fwAD.unpack_dual(dual_z).tangent)
            #     dual_z = self._layer_norm(dual_z)

            #     # second layer
            #     dual_act = block[-1](dual_z)
            #     dual_relu_act = self.act_fn.apply(dual_act)

            #     # peer normalization
            #     # if self.opt.model.peer_normalization > 0:
            #     #     peer_loss = self._calc_peer_normalization_loss(block_idx, dual_relu_act)
            #     #     scalar_outputs["Peer Normalization"] += peer_loss
            #     #     scalar_outputs["Loss"] += self.opt.model.peer_normalization * peer_loss

            #     ff_loss, ff_accuracy = self._calc_ff_loss(dual_relu_act, posneg_labels)
            #     scalar_outputs[f"loss_layer_{block_idx}"] = ff_loss
            #     scalar_outputs[f"ff_accuracy_layer_{block_idx}"] = ff_accuracy
            #     scalar_outputs["Loss"] += ff_loss

            #     jvp = fwAD.unpack_dual(scalar_outputs["Loss"]).tangent
            #     block_jvps.append(jvp)
            
            # x = dual_relu_act.detach()
            # x = self._layer_norm(x)

            # xs.append(block_xs.copy())
            # us.append(block_us.copy())
            # jvps.append(block_jvps.copy())

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

        z = inputs["prediction_sample"]
        if not self.opt.model.convolutional:
            z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z)

        input_classification_model = []

        with torch.no_grad():
            for block_idx, block in enumerate(self.model):

                for layer_idx, layer in enumerate(block):
                    z = layer(z)
                    z = self.bn[block_idx](z)
                    z = self.act_fn.apply(z)
                    z = self._layer_norm(z)

                if block_idx >= 1 or self.opt.model.num_blocks == 1:
                    input_classification_model.append(z.reshape(z.shape[0], -1))

                if self.opt.model.convolutional and (block_idx+1) in self.opt.model.conv.pool:
                    z = F.max_pool2d(z, 2, 2)  # maxpool

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

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, padding, bias=True):
        super(LocallyConnected2d, self).__init__()
        output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0], output_size[1], kernel_size**2)
        )
        self.bias = nn.Parameter(
            torch.randn(1, out_channels, output_size[0], output_size[1])
        )
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = (padding, padding, padding, padding)
        
    def forward(self, x):
        _, c, h, w = x.size()
        x = F.pad(x, self.padding, mode='constant', value=0)
        kh, kw = self.kernel_size
        dh, dw = self.stride

        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1]) + self.bias
        return out

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
        input = ctx.input
        assert input.shape == grad_output.shape
        grad_out = grad_output.clone()
        grad_out[input<0] = 0
        return grad_out
    
    @staticmethod
    def jvp(ctx, grad_input):
        input = ctx.input
        assert input.shape == grad_input.shape
        grad_in = grad_input.clone()
        grad_in[input<0] = 0
        return grad_in