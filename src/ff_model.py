import math
from functools import partial
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from src import utils

from pytorch_msssim import SSIM


class FF_model(torch.nn.Module):
    """The model trained with Forward-Forward (FF)."""

    def __init__(self, opt):
        super(FF_model, self).__init__()

        self.opt = opt
        self.act_fn = ReLU_full_grad()

        assert self.opt.model.num_blocks > 1, "model requires at least two blocks."

        self.model = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Initialiaze variables for dynamic threshold.
        self.threshold = self.opt.model.predictor_size * 0.625  # specific value should be adjusted
        self.epsilon = 0.9
        self.steps_done = 0

        if not self.opt.model.convolutional:
            self.num_channels = [getattr(self.opt.model.fully_connected, f"hidden_dim_{i+1}", 2000)
                                 for i in range(self.opt.model.num_blocks)]

            # Initialize the fully connectedmodel.
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

            prev_dimension = self.opt.input.input_channels
            self.ae_criterion = []
            self.convs = nn.ModuleList()

            for i in range(self.opt.model.num_blocks):

                # Initialize the convolutional model.
                if self.opt.model.conv.locally_connected:
                    self.model.append(nn.ModuleList([LocallyConnected2d(prev_dimension,  # in_channels
                                                                        getattr(self.opt.model.conv, f"channels_{i+1}", 1),     # out_channels
                                                                        getattr(self.opt.model.conv, f"output_size_{i+1}", 1),  # output_size
                                                                        getattr(self.opt.model.conv, f"kernel_size_{i+1}", 1),  # kernel_size
                                                                        getattr(self.opt.model.conv, f"stride_{i+1}", 1),       # stride
                                                                        getattr(self.opt.model.conv, f"padding_{i+1}", 1)       # padding
                                                                        )])) 
                else:
                    self.model.append(nn.ModuleList([nn.Conv2d(prev_dimension,                                       # in_channels
                                                            getattr(self.opt.model.conv, f"channels_{i+1}", 1),     # out_channels
                                                            getattr(self.opt.model.conv, f"kernel_size_{i+1}", 1),  # kernel_size
                                                            stride=getattr(self.opt.model.conv, f"stride_{i+1}", 1),       # stride
                                                            padding=getattr(self.opt.model.conv, f"padding_{i+1}", 1)       # padding
                                                            )]))
                self.bn.append(nn.BatchNorm2d(getattr(self.opt.model.conv, f"channels_{i+1}", 1)))

                # Initialize decoder for autoencoder purposes.
                if i not in self.opt.model.conv.pool:
                    self.decoders.append(nn.ConvTranspose2d(getattr(self.opt.model.conv, f"channels_{i+1}", 1),      # in_channels 
                                                        prev_dimension,                                         # out_channels
                                                        getattr(self.opt.model.conv, f"kernel_size_{i+1}", 1),  # kernel_size
                                                        stride=getattr(self.opt.model.conv, f"stride_{i+1}", 1),       # stride
                                                        padding=getattr(self.opt.model.conv, f"padding_{i+1}", 1)       # padding
                                                        ))
                else:  # transpose convolution to upsample by scale of 2, currently hard coded for 3*3 kernel
                    self.decoders.append(nn.ConvTranspose2d(getattr(self.opt.model.conv, f"channels_{i+1}", 1),      # in_channels
                                                            prev_dimension,                                         # out_channels
                                                            getattr(self.opt.model.conv, f"kernel_size_{i+1}", 1) + 1,  # kernel_size
                                                            stride=getattr(self.opt.model.conv, f"stride_{i+1}", 1) + 1,       # stride
                                                            padding=getattr(self.opt.model.conv, f"padding_{i+1}", 1)       # padding
                                                            ))
                self.ae_criterion.append(SSIM(data_range=1.0, size_average=True, channel=prev_dimension))
                
                # Initialize convolutional modules for calculating spatial goodness
                conv = nn.Conv2d(getattr(self.opt.model.conv, f"channels_{i+1}", 1), 1, 3, 1, 1)
                conv.weight = nn.parameter.Parameter(torch.ones_like(conv.weight) / (9 * getattr(self.opt.model.conv, f"channels_{i+1}", 1)))
                conv.weight.requires_grad = False
                conv.bias = nn.parameter.Parameter(torch.zeros_like(conv.bias))
                conv.bias.requires_grad = False
                self.convs.append(conv)

                prev_dimension = getattr(self.opt.model.conv, f"channels_{i+1}", 1)
        
        conv = nn.Conv2d(3, 1, 3, 1, 1)
        conv.weight = nn.parameter.Parameter(torch.ones_like(conv.weight) / (9 * 3))
        conv.weight.requires_grad = False
        conv.bias = nn.parameter.Parameter(torch.zeros_like(conv.bias))
        conv.bias.requires_grad = False
        self.convs.append(conv)

        # Initialize layer-wise predictor and avg-pooling layers for prediction loss & FF loss.
        self.pred_loss = nn.CrossEntropyLoss()
        self.layer_predictor = nn.Linear(opt.model.predictor_size, opt.input.num_classes)  # predictor for prediction loss
        self.goodness_predictor = nn.Linear(opt.model.predictor_size, 1)  # predictor for FF goodness
        if self.opt.model.convolutional:
            self.avgpools = nn.ModuleList()
            for i in range(len(self.num_channels)):
                reduce_factor = self.num_channels[i] // opt.model.predictor_size
                reduce_log2_factor = int(math.log2(reduce_factor))
                if reduce_log2_factor % 2 > 0:
                    avgpool = nn.AvgPool2d(kernel_size=(2**(reduce_log2_factor//2), 2**(reduce_log2_factor//2+1)))
                else:
                    avgpool = nn.AvgPool2d(kernel_size=(2**(reduce_log2_factor//2), 2**(reduce_log2_factor//2)))
                self.avgpools.append(avgpool)

        # Initialize AutoEncoder reconstrution loss.
        self.ae_loss = nn.MSELoss()

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
        if opt.training.downstream_method == "2..n":
            channels_for_classification_loss = sum(
                self.num_channels[-i] for i in range(1, self.opt.model.num_blocks)
            ) if self.opt.model.num_blocks > 1 else self.num_channels[0]
        else:  # "n", only use last block for classification
            if opt.model.convolutional and opt.model.num_blocks in opt.model.conv.pool: 
                channels_for_classification_loss = self.num_channels[-1]//4
            else:
                channels_for_classification_loss = self.num_channels[-1]

        self.linear_classifier = nn.Sequential(
            nn.Linear(channels_for_classification_loss, opt.input.num_classes, bias=False)
        )
        self.classification_loss = nn.CrossEntropyLoss()

        # Initialize weights.
        self._init_weights()

    def _init_weights(self):
        # Initialize encoder weights.
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
                        # this part of code needs to be checked before usage
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
                            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                        elif self.opt.training.init == "Xavier":
                            torch.nn.init.xavier_normal_(m.weight)
                        torch.nn.init.zeros_(m.bias)

        # Initialize decoder weights.
        for m in self.decoders.modules():
            if isinstance(m, nn.ConvTranspose2d):
                if self.opt.training.init == "He":
                    torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                elif self.opt.training.init == "Xavier":
                    torch.nn.init.normal_(
                        m.weight, mean=0, std=math.sqrt(1) / math.sqrt(m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3])
                    )
                torch.nn.init.zeros_(m.bias)

        # Initialize linear classifier weights.
        for m in self.linear_classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)

    def _layer_norm(self, z, layer_idx, eps=1e-8):
        z = z - torch.mean(z).detach()  # reduce mean for normalization
        if self.opt.model.norm_type == "layer":  # layer normalization
            scale = (torch.sqrt(torch.mean(z ** 2, dim=list(range(1,len(z.shape))), keepdim=True)) + eps) 
        elif self.opt.model.norm_type == "channel":  # channel normalization
            scale = (torch.sqrt(torch.mean(z ** 2, dim=list(range(2,len(z.shape))), keepdim=True)) + eps) 
        elif self.opt.model.norm_type == "divisive":  # divisive normalization
            scale = (torch.sqrt(self.convs[layer_idx](z ** 2)) + eps) 
        # print(torch.min(scale), torch.max(scale))
        return z / scale
    
    def _calc_ae_loss(self, x, z, start_layer_idx, end_layer_idx):
        # Calculate AutoEncoder Loss
        reconstruct_x = z.clone()
        for layer_idx in range(end_layer_idx, start_layer_idx-1, -1):
            reconstruct_x = self.decoders[layer_idx](reconstruct_x)

            if layer_idx > start_layer_idx:
                reconstruct_x = self.act_fn.apply(reconstruct_x)
                reconstruct_x = self._layer_norm(reconstruct_x, layer_idx-1)
            
            else:
                reconstruct_x = torch.sigmoid(reconstruct_x)
                ae_loss = 1 - self.ae_criterion[start_layer_idx](reconstruct_x, x)
                mse_loss = self.ae_loss(reconstruct_x, x)
                return ae_loss + mse_loss

    def _calc_peer_normalization_loss(self, idx, z):
        # Calculate peer normalization Loss. Only calculate mean activity over positive samples.
        mean_activity = torch.mean(z[:self.opt.input.batch_size], dim=0)

        self.running_means[idx] = self.running_means[
            idx
        ].detach() * self.opt.model.momentum + mean_activity * (
            1 - self.opt.model.momentum
        )

        peer_loss = (torch.mean(self.running_means[idx]) - self.running_means[idx]) ** 2
        return torch.mean(peer_loss)

    def _calc_ff_loss(self, z, labels, layer_index):
        # Calculate FF Loss
        if self.opt.model.goodness_type == "spatial":
            # spatial goodness
            sum_of_squares = self.convs[layer_index](z ** 2)
            logits = torch.mean(sum_of_squares, dim=list(range(1,len(sum_of_squares.shape))))

        elif self.opt.model.goodness_type == "revised_spatial":
            # revised spatial goodness
            z = self.convs[layer_index](z)
            sum_of_squares = torch.sum(z ** 2, dim=list(range(1,len(z.shape))))
            logits = sum_of_squares - (z.reshape(len(z),-1).shape[1] * 0.625)

        elif self.opt.model.goodness_type == "flatten":
            # flatten (naive) goodness
            sum_of_squares = torch.sum(z ** 2, dim=list(range(1,len(z.shape))))
            logits = torch.sum(z ** 2, dim=list(range(1,len(z.shape))))
            
        elif self.opt.model.goodness_type == "avgpool":
            # avgpool goodness
            z = self.avgpools[layer_index](z)
            z = z.reshape(z.shape[0], -1)

            sum_of_squares = torch.sum(z ** 2, dim=list(range(1,len(z.shape))))
            logits = torch.sum(z ** 2, dim=list(range(1,len(z.shape))))

        elif self.opt.model.goodness_type == "fully_connected":
            # fully connected goodness
            z = self.avgpools[layer_index](z)
            z = z.reshape(z.shape[0], -1)

            logits = self.goodness_predictor(z)
            logits = logits.squeeze(-1)


        if self.opt.model.threshold_type == "dynamic":
            # dynamic threshold
            self.epsilon = 0.9 * math.exp(-1. * self.steps_done / 10000)
            self.threshold = (1 - self.epsilon) * self.threshold + self.epsilon * torch.mean(logits).detach()
            self.steps_done += 1

            logits = logits - self.threshold
        
        elif self.opt.model.threshold_type == "fixed":
            # fixed threshold
            logits = logits - self.threshold

        elif self.opt.model.threshold_type == "mean":
            # mean as threshold
            logits = logits - mean_logits

        elif self.opt.model.threshold_type == "mean_detached":
            # detached mean as threshold
            mean_logits = torch.mean(logits).detach()
            logits = logits - mean_logits


        # Calculate FF Loss
        ff_loss = self.ff_loss(logits, labels.float())

        # Calculate FF Accuracy
        with torch.no_grad():
            ff_accuracy = (
                torch.sum((torch.sigmoid(logits) > 0.5) == labels)
                / z.shape[0]
            ).item()
        return ff_loss, ff_accuracy
    
    def _calc_pred_loss(self, z, labels, layer_index):
        # Calculate Prediction Loss
        if self.opt.model.convolutional:
            z = self.avgpools[layer_index](z)
            z = z.reshape(z.shape[0], -1)

        z_predicted = self.layer_predictor(z)
        pred_loss = self.pred_loss(z_predicted, labels)

        with torch.no_grad():
            pred_accuracy = (
                torch.sum(torch.argmax(z_predicted, dim=1) == labels)
                / z.shape[0]
            ).item()
        return pred_loss, pred_accuracy
    
    def _construct_neg_sample(self, sample):
        # use Autoencoder to construct negative sample
        with torch.no_grad():
            # encoder
            z = sample.clone()
            for block_idx, block in enumerate(self.model):
                z = block[0](z)
                # z = self.bn[block_idx](z)
                z = self.act_fn.apply(z)
                z = self._layer_norm(z, block_idx)

                if (block_idx+1) < self.opt.model.num_blocks:
                    if self.opt.model.convolutional and (block_idx+1) in self.opt.model.conv.pool:
                        z = F.max_pool2d(z, 2, 2)  # maxpool

            # decoder
            for layer_idx in range(self.opt.model.num_blocks-1, -1, -1):
                z = self.decoders[layer_idx](z)

                if layer_idx > 0:
                    # z = self.bn[layer_idx-1](z)
                    z = self.act_fn.apply(z)
                    z = self._layer_norm(z, layer_idx-1)
                else:
                    z = torch.sigmoid(z)
            return z

    def forward(self, inputs, labels):
        # Main forward function for the model.
        scalar_outputs = {
            "Loss": torch.zeros(1, device=self.opt.device),
            "Peer Normalization": torch.zeros(1, device=self.opt.device),
        }

        if self.opt.training.sim_pred.beta == 1:  # FF only
            x = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0)  # original FF
            # x = torch.cat([inputs["original_sample"], self._construct_neg_sample(inputs["original_sample"])], dim=0)  # negative sample from AE

            posneg_labels = torch.zeros(x.shape[0], device=self.opt.device)  
            posneg_labels[self.opt.input.batch_size:] = 1  # for minimizing goodness for positive samples

        elif self.opt.training.sim_pred.beta == 0:  # prediction only
            x = inputs["original_sample"]

        else:  # FF + prediction
            # Concatenate positive and negative samples and create corresponding labels.
            ff_x = torch.cat([inputs["pos_images"], inputs["neg_images"]], dim=0)
            posneg_labels = torch.zeros(ff_x.shape[0], device=self.opt.device)
            posneg_labels[self.opt.input.batch_size:] = 1  # for minimizing goodness for positive samples

            # Prediction sample
            pred_x = inputs["original_sample"]

            # Comcatenate samples for FF and pred
            x = torch.cat([ff_x, pred_x], dim=0)

        xs,us,jvps = [],[],[]

        if not self.opt.model.convolutional:
            x = x.reshape(x.shape[0], -1)
        x = self._layer_norm(x, -1)
            
        # reconstruction_prev_layer_num = 0
        # reconstruction_targets = x.clone()

        for block_idx, block in enumerate(self.model):

            # block_xs, block_us, block_jvps = [],[],[]

            # if block_idx == 0 or block_idx in self.opt.model.reconstruction_objectives:
            #     reconstruction_targets = x.detach().clone()
            #     reconstruction_prev_layer_num = block_idx

            z = x.clone()
            if self.opt.model.convolutional and block_idx in self.opt.model.conv.pool:
                z = F.max_pool2d(z, 2, 2)  # maxpool
            z = block[0](z)
            # z = self.bn[block_idx](z)
            z = self.act_fn.apply(z)
            if self.opt.training.dropout > 0:
                z = F.dropout(z, p=self.opt.training.dropout, training=True)
                
            if block_idx + 1 in self.opt.model.reconstruction_objectives:

                if 0 < self.opt.training.sim_pred.beta < 1:  # separate FF and pred inputs for simpred loss calculation
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

                    # FF loss (AutoEncoder loss commented out for now)
                    if self.opt.training.sim_pred.beta < 1:
                        ff_loss, ff_accuracy = self._calc_ff_loss(ff_z, posneg_labels, block_idx)
                        # ae_loss = self._calc_ae_loss(reconstruction_targets[:self.opt.input.batch_size], ff_z[:self.opt.input.batch_size], reconstruction_prev_layer_num, block_idx)
                    else:
                        ff_loss, ff_accuracy = self._calc_ff_loss(z, posneg_labels, block_idx)  # for FF only
                        # ae_loss = self._calc_ae_loss(reconstruction_targets[:self.opt.input.batch_size], z[:self.opt.input.batch_size], reconstruction_prev_layer_num, block_idx)
                    scalar_outputs[f"loss_layer_{block_idx}"] = ff_loss
                    scalar_outputs[f"ff_accuracy_layer_{block_idx}"] = ff_accuracy
                    scalar_outputs["Loss"] += self.opt.training.sim_pred.beta * ff_loss

                    # scalar_outputs[f"ae_loss_layer_{block_idx}"] = ae_loss
                    # scalar_outputs["Loss"] += ae_loss

                x = z.detach()
            else:
                x = z.clone()
            x = self._layer_norm(x, block_idx)



            # code for forward gradients, for future implementation

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

        # Downstream Classification
        scalar_outputs = self.forward_downstream_classification_model(
            inputs, labels, scalar_outputs=scalar_outputs
        )

        return scalar_outputs, xs, us, jvps

    def forward_downstream_classification_model(
        # Function for the downstream classification model.
        self, inputs, labels, scalar_outputs=None,
    ):
        if scalar_outputs is None:
            scalar_outputs = {
                "Loss": torch.zeros(1, device=self.opt.device),
            }

        # z = inputs["original_sample"]
        z = inputs["neutral_sample"]
        if not self.opt.model.convolutional:
            z = z.reshape(z.shape[0], -1)
        z = self._layer_norm(z, -1)

        input_classification_model = []

        with torch.no_grad():
            for block_idx, block in enumerate(self.model):

                for layer_idx, layer in enumerate(block):
                    z = layer(z)
                    # z = self.bn[block_idx](z)
                    z = self.act_fn.apply(z)

                if self.opt.training.downstream_method == "2..n" and block_idx >= 1:
                    input_classification_model.append(z.reshape(z.shape[0], -1))

                z = self._layer_norm(z, block_idx)

                if self.opt.model.convolutional and (block_idx+1) in self.opt.model.conv.pool:
                    z = F.max_pool2d(z, 2, 2)  # maxpool

        if self.opt.training.downstream_method == "2..n":
            input_classification_model = torch.concat(input_classification_model, dim=-1)
        else:
            input_classification_model = z.reshape(z.shape[0], -1)

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