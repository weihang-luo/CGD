import copy
import functools
import os
import time
import numpy as np
from omegaconf import OmegaConf
import torch
import logging
from pathlib import Path
import torch.nn.functional as F

import blobfile as bf
import torch as th
from torch.optim import AdamW

from .fp16_util import JointMixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from guided_diffusion.resample import create_named_schedule_sampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        joint_model,
        diffusion,
        data,
        batch_size,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        checkpoint_dir,
        joint_checkpoint_dir,
        resume_checkpoint,
        joint_resume_checkpoint,
        sample_num,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        mask_crop=0,
    ):
        self.model = model
        self.joint_model = joint_model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        self.joint_checkpoint_dir = joint_checkpoint_dir
        self.resume_checkpoint = resume_checkpoint
        self.joint_resume_checkpoint = joint_resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.device = next(self.model.parameters()).device
        self.sample_num = sample_num
        self.mask_crop = mask_crop
        

        self.step = 0
        self.resume_step = 0
        
        c_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger = get_logger(
                        bf.join(os.path.dirname(self.checkpoint_dir), 
                                f"JointTrain_logger_from_{'start' if not resume_checkpoint else 'resume'}_{c_time}.log")
                                )
        self.logger.info(OmegaConf.to_yaml(self.diffusion.conf))
        # self.sync_cuda = th.cuda.is_available()
        if self.resume_checkpoint:
            self._load_parameters()

        self.mp_trainer = JointMixedPrecisionTrainer(
            model=self.model,
            joint_model=self.joint_model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        self.joint_opt = AdamW(
            self.mp_trainer.joint_master_params, lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
            self.joint_ema_params = [
                self._load_ema_parameters(rate, joint=True) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]
            self.joint_ema_params = [
                copy.deepcopy(self.mp_trainer.joint_master_params)
                for _ in range(len(self.ema_rate))
            ]

    def _load_parameters(self):
        resume_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'model')
        if resume_checkpoint:
            self.logger.info(f"loading resume model from checkpoint: {resume_checkpoint}")
            self.resume_step = parse_resume_step_from_filename(str(resume_checkpoint))
            self.model.load_state_dict(torch.load(resume_checkpoint))

        joint_resume_checkpoint = find_resume_checkpoint(self.joint_resume_checkpoint, 'model')
        if joint_resume_checkpoint:
            self.logger.info(f"loading resume joint_model from checkpoint: {joint_resume_checkpoint}")
            # self.resume_step = parse_resume_step_from_filename(str(joint_crop_resume_checkpoint))
            self.joint_model.load_state_dict(torch.load(joint_resume_checkpoint))


    def _load_ema_parameters(self, rate, joint=False):
        if joint:
            ema_params = copy.deepcopy(self.mp_trainer.joint_master_params)
            main_checkpoint = find_resume_checkpoint(self.joint_resume_checkpoint, 'model')
        else:
            ema_params = copy.deepcopy(self.mp_trainer.master_params)
            main_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'model')

        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            # logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            self.logger.info(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = torch.load(ema_checkpoint)
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict, joint=joint)

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint(self.resume_checkpoint, 'model')
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            # logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            self.logger.info(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            self.opt.load_state_dict(torch.load(opt_checkpoint))

        joint_main_checkpoint = find_resume_checkpoint(self.joint_resume_checkpoint, 'model')
        joint_opt_checkpoint = bf.join(
            bf.dirname(joint_main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(joint_opt_checkpoint):
            # logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            self.logger.info(f"loading joint_optimizer state from joint_checkpoint: {joint_opt_checkpoint}")
            self.joint_opt.load_state_dict(torch.load(joint_opt_checkpoint))

    def run_loop(self):
        self.logger.info(f"training start! step:[{self.step + self.resume_step}]")
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch = next(self.data)
            self.run_step(batch)
            if self.step % self.save_interval == 0:
                self.save()
                allocated, peak_allocated = check_memory_usage()
                self.logger.info(f"Allocated memory: {allocated / 1024**2:.2f} MB, Peak allocated memory: {peak_allocated / 1024**2:.2f} MB")
                # sample
                if self.step != 0:
                    self.logger.info(f"training checkpoint step[{self.step + self.resume_step}] sampling...")
                    self.diffusion.train_sample_fn(self.step + self.resume_step, self.model, bf.join(self.checkpoint_dir, 'output'), self.sample_num)
                    self.diffusion.train_sample_fn(self.step + self.resume_step, self.joint_model, bf.join(self.joint_checkpoint_dir, 'output'), self.sample_num)
                    self.logger.info(f"generate {self.sample_num * 4} samples per model!")
                # Run for a finite amount of time in integration tests.
                # if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                #     return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0 and self.step != 0:
            self.save()
            
            

    def run_step(self, batch):
        self.forward_backward(batch) 
        took_step = self.mp_trainer.optimize(self.opt, self.joint_opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        # self.log_step()

    def forward_backward(self, batch):
        self.mp_trainer.zero_grad()
        

        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(self.device)

        x = batch['x']
        label = batch['label']
        crop = batch['crop']
        crop_label = batch['crop_label']
        pos = batch['pos']

        # mask = th.ones_like(x)

        # batch 中存在不同图像的crop位置不同 改代码不适用 暂不修改 忽略
        # if self.mask_crop:
        #     print(pos, pos[1])
        #     mask[:,:,pos[1]:pos[1]+crop.shape[-1], pos[0]:pos[0]+crop.shape[-1]] = 0

        x = x.to(self.device)
        label = label.to(self.device)
        crop = crop.to(self.device)
        crop_label = crop_label.to(self.device)
        # mask = mask.to(self.device)

        t, weights = self.schedule_sampler.sample(x.shape[0], x.device)
        model_kwargs={}
        model_kwargs['y']=label
        model_kwargs['crop_y']=crop_label
        model_kwargs['pos']=pos

        # model_kwargs['mask']= mask

        compute_losses = functools.partial(
            self.diffusion.joint_training_losses,
            self.model,
            self.joint_model,
            x,
            crop,
            t,
            model_kwargs=model_kwargs,
        )
        losses = compute_losses()

        loss_main = losses["loss"]

        loss_crop = losses["joint_mse"]

        loss_joint = loss_main + loss_crop


        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, loss_joint.detach()
            )
        loss = (loss_joint * weights).mean()
        # log_loss_dict(
        #     self.diffusion, t, {k: v * weights for k, v in losses.items()}
        # )
        if self.step % self.log_interval == 0:
        #     logger.dumpkvs()
            loss1 = (loss_main.detach() * weights).mean()
            loss2 = (loss_crop.detach() * weights).mean()
            self.logger.info(f"step:[{self.step + self.resume_step}],"\
                             f"loss[main/cropmse]:{loss:.6f}[{loss1:.6f}/{loss2:.6f}] ")
            # self.check_memory_usage()
        self.mp_trainer.backward(loss)
    
    def _update_ema(self):
        for rate, params,joint_params in zip(self.ema_rate, self.ema_params, self.joint_ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)
            update_ema(joint_params, self.mp_trainer.joint_master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group, joint_param_group in zip(self.opt.param_groups, self.opt.joint_param_groups):
            param_group["lr"] = lr
            joint_param_group["lr"] = lr

    # def log_step(self):
    #     logger.logkv("step", self.step + self.resume_step)
    #     logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params, joint):
            state_dict = self.mp_trainer.master_params_to_state_dict(params, joint=joint)
            # logger.log(f"saving model rate:{rate}...")
            self.logger.info(f"saving {'Joint' if joint else ''} model rate:[{rate}]...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            cp_dir = self.joint_checkpoint_dir if joint else self.checkpoint_dir
            with bf.BlobFile(bf.join(cp_dir, filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params, joint=False)
        save_checkpoint(0, self.mp_trainer.joint_master_params, joint=True)
        for rate, params, joint_params in zip(self.ema_rate, self.ema_params, self.joint_ema_params):
            save_checkpoint(rate, params, joint=False)
            save_checkpoint(rate, joint_params, joint=True)

        filename = f"opt{(self.step+self.resume_step):06d}.pt"
        with bf.BlobFile(bf.join(self.checkpoint_dir, filename),"wb") as f:
            th.save(self.opt.state_dict(), f)

        with bf.BlobFile(bf.join(self.joint_checkpoint_dir, filename),"wb") as f:
            th.save(self.joint_opt.state_dict(), f)



def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


# def get_blob_logdir():
#     # You can change this to be a separate path to save checkpoints to
#     # a blobstore or some external drive.
#     return logger.get_dir()


def find_resume_checkpoint(dirname, key):
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    dirname = Path(dirname)
    assert dirname.exists() ,'checkpoint dir not exists'

    ckt_models = [f for f in dirname.glob(f'{key}*.pt')]

    if ckt_models:
        ckt_models.sort()
        last_model_name = ckt_models[-1]
        return last_model_name
    
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


# def log_loss_dict(diffusion, ts, losses):
#     for key, values in losses.items():
#         logger.logkv_mean(key, values.mean().item())
#         # Log the quantiles (four quartiles, in particular).
#         for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
#             quartile = int(4 * sub_t / diffusion.num_timesteps)
#             logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

def check_memory_usage():
    allocated = torch.cuda.memory_allocated()
    peak_allocated = torch.cuda.max_memory_allocated()
    return allocated, peak_allocated
    self.logger.info(f"Allocated memory: {allocated / 1024**2:.2f} MB, Peak allocated memory: {peak_allocated / 1024**2:.2f} MB")

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return lr

def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = th.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)

class ClassifierTrainLoop:

    def __init__(
        self,
        model,
        diffusion,
        data,
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        classifier_use_fp16=False,
        schedule_sampler=None,
        checkpoint="",
        resume_checkpoint="",
        log_interval=100,
        save_interval=10000,         
    ):
        assert schedule_sampler , 'schedule_sampler is None'
        self.schedule_sampler = schedule_sampler
        self.checkpoint = checkpoint
        self.resume_checkpoint = resume_checkpoint
        self.noised = noised
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.lr = lr
        self.anneal_lr = anneal_lr
        self.now_lr = lr
        self.classifier_use_fp16 = classifier_use_fp16
        self.iterations = iterations
        self.device = next(self.model.parameters()).device
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.weight_decay = weight_decay
            
        c_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        os.makedirs(self.checkpoint, exist_ok=True)
        self.logger = get_logger(
                        bf.join(self.checkpoint, 
                                f"Classifier_Train_logger_{'start' if not resume_checkpoint else 'resume'}_{c_time}.log")
                                )
        
        self.resume_step = 0
        if self.resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            self.logger.info(f"loading checkpoint: {self.resume_checkpoint}...")
            self.model.load_state_dict(torch.load(resume_checkpoint))

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.classifier_use_fp16,
            initial_lg_loss_scale=16.0,
        )

        self.logger.info(f"creating optimizer...")
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_checkpoint:
            opt_checkpoint = bf.join(
                bf.dirname(self.resume_checkpoint), f"classifier_opt{self.resume_step:06}.pt"
            )
            self.logger.info(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            self.opt.load_state_dict(
                torch.load(opt_checkpoint)
            )
        
    def run_loop(self):
        self.step = 0
        self.logger.info(f"training start! step:[{self.step + self.resume_step}]")
        
        while (
            self.step + self.resume_step < self.iterations 
        ):
            if self.anneal_lr:
                self.now_lr = set_annealed_lr(self.opt, self.lr, (self.step + self.resume_step) / self.iterations)
            
            batch, cond = next(self.data)
            self.forward_backward_log(batch, cond)
            self.mp_trainer.optimize(self.opt)

            # save model
            if self.step % self.save_interval == 0:
                self.save()
            self.step = self.step + 1



    def forward_backward_log(self, batch, cond, prefix='train'):

        labels = cond.to(self.device)
        batch = batch.to(self.device)
        # Noisy images
        if self.noised:
            t, _ = self.schedule_sampler.sample(batch.shape[0], self.device)
            batch = self.diffusion.q_sample(batch, t)
        else:
            t = th.zeros(batch.shape[0], dtype=th.long, device=self.device)

        self.mp_trainer.zero_grad()

        logits = self.model(batch, timesteps=t)
        loss = F.cross_entropy(logits, labels, reduction="none")
        loss = loss.mean()

        if self.step % self.log_interval == 0:

            acc_1 = compute_top_k(
                logits, labels, k=1, reduction="mean"
            )
            acc_3 = compute_top_k(
                logits, labels, k=3, reduction="mean"
            )
            self.logger.info(f"step:[{self.step + self.resume_step}],loss:[{loss:.6f}],acc_1:[{acc_1:.6f}],acc_3:[{acc_3:.6f}] ,learning rate:[{self.now_lr}/{self.lr}]")
        self.mp_trainer.backward(loss)

    def save(self):
        self.logger.info(f"saving model step{self.step + self.resume_step}...")
        allocated, peak_allocated = check_memory_usage()
        self.logger.info(f"Allocated memory: {allocated / 1024**2:.2f} MB, Peak allocated memory: {peak_allocated / 1024**2:.2f} MB")
        self.logger.info(f" base_lr:{self.lr}, learning rate:{self.now_lr:.6f}")
        th.save(
            self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params),
            os.path.join(self.checkpoint, f"classifier_model{(self.step + self.resume_step):06d}.pt"),
        )
        th.save(self.opt.state_dict(), os.path.join(self.checkpoint, f"classifier_opt{(self.step + self.resume_step):06d}.pt"))



class TrainLoopSampleClass:

    def __init__(
        self,
        model,
        data,
        iterations=150000,
        lr=1e-4,
        weight_decay=1e-3,
        anneal_lr=False,
        use_fp16=False,
        checkpoint="",
        resume_checkpoint="",
        log_interval=100,
        save_interval=10000,         
    ):
        self.checkpoint = checkpoint
        self.resume_checkpoint = resume_checkpoint
        self.model = model
        self.data = data
        self.lr = lr
        self.anneal_lr = anneal_lr
        self.now_lr = lr
        self.use_fp16 = use_fp16
        self.iterations = iterations
        self.device = next(self.model.parameters()).device
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.weight_decay = weight_decay
            
        c_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        os.makedirs(self.checkpoint, exist_ok=True)
        self.logger = get_logger(
                        bf.join(self.checkpoint, 
                                f"SampleClass_Train_log_{'start' if not resume_checkpoint else 'resume'}_{c_time}.log")
                                )
        
        self.resume_step = 0
        if self.resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)  # 含model
            self.logger.info(f"loading checkpoint: {self.resume_checkpoint}...")
            self.model.load_state_dict(torch.load(resume_checkpoint))

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            initial_lg_loss_scale=16.0,
        )

        self.logger.info(f"creating optimizer...")
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_checkpoint:
            opt_checkpoint = bf.join(
                bf.dirname(self.resume_checkpoint), f"sampleclass_opt{self.resume_step:06}.pt"
            )
            self.logger.info(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            self.opt.load_state_dict(
                torch.load(opt_checkpoint)
            )
        
    def run_loop(self):
        self.step = 0
        self.logger.info(f"training start! step:[{self.step + self.resume_step}]")
        
        while (
            self.step + self.resume_step < self.iterations 
        ):
            if self.anneal_lr:
                self.now_lr = set_annealed_lr(self.opt, self.lr, (self.step + self.resume_step) / self.iterations)
            
            batch, cond = next(self.data)
            self.forward_backward_log(batch, cond)
            self.mp_trainer.optimize(self.opt)

            # save model
            if self.step % self.save_interval == 0:
                self.save()
            self.step = self.step + 1



    def forward_backward_log(self, batch, cond, prefix='train'):

        labels = cond.to(self.device)
        batch = batch.to(self.device)
        # Noisy images

        self.mp_trainer.zero_grad()

        logits = self.model(batch)
        loss = F.cross_entropy(logits, labels, reduction="mean")

        if self.step % self.log_interval == 0:

            acc_1 = compute_top_k(
                logits, labels, k=1, reduction="mean"
            )
            acc_5 = compute_top_k(
                logits, labels, k=5, reduction="mean"
            )
            self.logger.info(f"step:[{self.step + self.resume_step}],loss:[{loss:.6f}],acc_1:[{acc_1:.6f}],acc_5:[{acc_5:.6f}] ,learning rate:[{self.now_lr}/{self.lr}]")
        self.mp_trainer.backward(loss)

    def save(self):
        self.logger.info(f"saving model step{self.step + self.resume_step}...")
        allocated, peak_allocated = check_memory_usage()
        self.logger.info(f"Allocated memory: {allocated / 1024**2:.2f} MB, Peak allocated memory: {peak_allocated / 1024**2:.2f} MB")
        self.logger.info(f" base_lr:{self.lr}, learning rate:{self.now_lr:.6f}")
        th.save(
            self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params),
            os.path.join(self.checkpoint, f"sampleclass_model{(self.step + self.resume_step):06d}.pt"),
        )
        th.save(self.opt.state_dict(), os.path.join(self.checkpoint, f"sampleclass_opt{(self.step + self.resume_step):06d}.pt"))



