if __name__ == "__main__":
    import os
    import pathlib
    import sys

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import copy
import os
import pathlib
import random

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace


OmegaConf.register_new_resolver("eval", eval, replace=True)


class _NoOpWandbRun:
    def log(self, *args, **kwargs):
        pass


class TrainFastResidualWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: BaseImagePolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: BaseImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=trainable_params)
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.resume:
            latest_ckpt_path = self.get_checkpoint_path()
            if latest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {latest_ckpt_path}")
                self.load_checkpoint(path=latest_ckpt_path)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs
            ) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel = None
        if self.ema_model is not None:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        if str(cfg.logging.mode).lower() == "disabled":
            wandb_run = _NoOpWandbRun()
        else:
            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging,
            )
            wandb.config.update({"output_dir": self.output_dir})

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk,
        )

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                step_log = {}
                train_losses = []
                self.model.train()
                with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Fast residual training epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                    ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        do_optimizer_step = (
                            (batch_idx + 1) % cfg.training.gradient_accumulate_every == 0
                            or batch_idx == (len(train_dataloader) - 1)
                        )
                        if do_optimizer_step:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if ema is not None and do_optimizer_step:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (
                                cfg.training.max_train_steps is not None
                                and batch_idx >= cfg.training.max_train_steps - 1):
                            break

                step_log["train_loss"] = float(np.mean(train_losses))

                eval_policy = self.ema_model if self.ema_model is not None else self.model
                eval_policy.eval()

                if self.epoch % cfg.training.val_every == 0:
                    val_losses = []
                    with torch.no_grad():
                        with tqdm.tqdm(
                                val_dataloader,
                                desc=f"Fast residual validation epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                            ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss.item())
                                if (
                                        cfg.training.max_val_steps is not None
                                        and batch_idx >= cfg.training.max_val_steps - 1):
                                    break
                    if len(val_losses) > 0:
                        step_log["val_loss"] = float(np.mean(val_losses))

                if self.epoch % cfg.training.sample_every == 0 and train_sampling_batch is not None:
                    with torch.no_grad():
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        pred = eval_policy.predict_action(batch["obs"])["action_pred"]
                        target = batch["action"]
                        if target.shape[1] != pred.shape[1]:
                            target = target[:, -pred.shape[1]:]
                        mse = torch.nn.functional.mse_loss(pred, target)
                        step_log["train_action_mse_error"] = mse.item()

                if self.epoch % cfg.training.checkpoint_every == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        if cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name="residual_policy/mlp",
)
def main(cfg):
    workspace = TrainFastResidualWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
