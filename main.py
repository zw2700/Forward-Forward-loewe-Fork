import time
from collections import defaultdict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src import utils

import wandb


def train(opt, model, optimizer):
    start_time = time.time()
    train_loader = utils.get_data(opt, "train")
    num_steps_per_epoch = len(train_loader)
    # return model

    for epoch in range(opt.training.epochs):
        train_results = defaultdict(float)
        optimizer = utils.update_learning_rate(optimizer, opt, epoch)

        for inputs, labels in train_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            optimizer.zero_grad()

            scalar_outputs, xs, us, jvps = model(inputs, labels)

            # forward gradients
            # for block_idx in range(opt.model.num_blocks):
            #     for layer_idx in range(opt.model.num_layers_per_block - 1):
            #         x,u,jvp = xs[block_idx][layer_idx],us[block_idx][layer_idx],jvps[block_idx][layer_idx]
            #         w_grad = torch.matmul(u.T,x)*jvp
            #         b_grad = torch.matmul(u.T,torch.ones(x.shape[0], device=opt.device))*jvp
            #         model.model[block_idx][layer_idx].weight.grad = w_grad
            #         model.model[block_idx][layer_idx].bias.grad = b_grad
            
            # print(optimizer.param_groups[0]["params"][1].grad, optimizer.param_groups[0]["params"][3].grad)

            # backward gradients for final layers in each block
            scalar_outputs["Loss"].backward()

            # for p in optimizer.param_groups[0]["params"]:
            #     if p.grad is not None:
            #         print(p.grad.shape, torch.linalg.norm(p.grad))


            optimizer.step()

            train_results = utils.log_results(
                train_results, scalar_outputs, num_steps_per_epoch
            )
            return model

        utils.print_results("train", time.time() - start_time, train_results, epoch)
        if opt.wandb.activate:
            wandb.log({"train": train_results}, step=epoch)
        start_time = time.time()

        # Validate.
        if epoch % opt.training.val_idx == 0 and opt.training.val_idx != -1:
            validate(opt, model, epoch=epoch)
        
        # return model

    return model


def validate(opt, model, epoch=None):
    test_time = time.time()
    test_results = defaultdict(float)

    data_loader = utils.get_data(opt, "val")
    num_steps_per_epoch = len(data_loader)

    model.eval()
    print("val")
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = utils.preprocess_inputs(opt, inputs, labels)

            scalar_outputs = model.forward_downstream_classification_model(
                inputs, labels
            )
            
            test_results = utils.log_results(
                test_results, scalar_outputs, num_steps_per_epoch
            )

    utils.print_results("val", time.time() - test_time, test_results, epoch=epoch)
    if epoch is not None and opt.wandb.activate:
        wandb.log({"val": test_results}, step=epoch)
    model.train()


@hydra.main(config_path=".", config_name="config", version_base=None)
def my_main(opt: DictConfig) -> None:
    opt = utils.parse_args(opt)
    if opt.wandb.activate:
        wandb.login(key=opt.wandb.key)
        wandb.init(project=opt.wandb.project, entity=opt.wandb.entity, name=opt.wandb.name)
    print(OmegaConf.to_yaml(opt))
    model, optimizer = utils.get_model_and_optimizer(opt)
    model = train(opt, model, optimizer)
    # validate(opt, model)

    # torch.save(model.state_dict(), opt.path_to_model)  # save model


if __name__ == "__main__":
    my_main()
