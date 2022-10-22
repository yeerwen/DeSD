# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import models as torchvision_models
import utils
import models.res3d as res3d
from models.res3d import DINOHead
from data_loader_ssl import Dataset3D
import shutil

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DeSD', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='res3d18', type=str, help="""Name of architecture to train.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of the DeSD head output.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DeSD head.
        Not normalizing leads to better performance but can make the training unstable.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature.')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.5, 0.7),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=0, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.""")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.8, 1.0),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/media/userdisk2/jpzhang/data_SSL/patches_v2/', type=str,
        help='Please specify path to the DeepLesion training data.')
    parser.add_argument('--list_path', default='Spacing_3D_update.txt', type=str)
    parser.add_argument('--output_dir', default="snapshots/DeSD_resnet50_300", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=50, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser

def train_DeSD(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # copy the key files
    os.makedirs(os.path.join(args.output_dir, "code"), exist_ok=True)
    shutil.copyfile("main_DeSD_ssl.py", os.path.join(args.output_dir, "code", "main_DeSD_ssl.py"))
    shutil.copyfile("utils.py", os.path.join(args.output_dir, "code", "utils.py"))
    shutil.copyfile("data_loader_ssl.py", os.path.join(args.output_dir, "code", "data_loader_ssl.py"))
    shutil.copyfile("models/res3d.py", os.path.join(args.output_dir, "code", "res3d.py"))
    shutil.copyfile("run_ssl.sh", os.path.join(args.output_dir, "code", "run_ssl.sh"))
    print("copy files finish!")

    # ============ preparing data ... ============
    train_set = Dataset3D(args.data_path, args.list_path, global_crop_size=(16, 96, 96), local_crop_size=(16, 48, 48), local_crops_number=args.local_crops_number)

    sampler = torch.utils.data.DistributedSampler(train_set, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        train_set,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )


    print(f"Data loaded: there are {len(train_set)} images.")

    # ============ building student and teacher networks ... ============
    if args.arch in res3d.__dict__.keys():
        student = res3d.__dict__[args.arch]()
        teacher = res3d.__dict__[args.arch+"_tea"]()

        print("Chossing Res3D as backbone....")
    else:
        print(f"Unknow architecture: {args.arch}")

    # 256, 512, 1024, and 2048 are the output dimensions of sub-encoders
    student = utils.MultiCropWrapper_v5(student,
        [DINOHead(256, args.out_dim, args.use_bn_in_head),
        DINOHead(512, args.out_dim, args.use_bn_in_head),
        DINOHead(1024, args.out_dim, args.use_bn_in_head),
        DINOHead(2048, args.out_dim, args.use_bn_in_head)])
    teacher = utils.MultiCropWrapper_v5_tea(teacher,
        [DINOHead(2048, args.out_dim, args.use_bn_in_head)])


    print(student)
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu], find_unused_parameters=True)
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu], find_unused_parameters=True)
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    desd_loss = DeSDLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        desd_loss=desd_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs): #args.epochs
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DeSd ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, desd_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'desd_loss': desd_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        total_time = time.time() - start_time
        start_time = time.time()
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, desd_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):

    now_it = 0
    average_loss = {"total_loss":[], "loss_ssl":[], "deep_loss_1":[],  "deep_loss_2":[],  "deep_loss_3":[]}

    for images in data_loader:
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + now_it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        with torch.cuda.amp.autocast(fp16_scaler is not None):

            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss, loss_ssl, self_loss_1, self_loss_2, self_loss_3 = desd_loss(student_output, teacher_output, epoch)
            average_loss["total_loss"].append(loss.item())
            average_loss["loss_ssl"].append(loss_ssl.item())
            average_loss["deep_loss_1"].append(self_loss_1.item())
            average_loss["deep_loss_2"].append(self_loss_2.item())
            average_loss["deep_loss_3"].append(self_loss_3.item())


        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()

        param_norms = None
        if fp16_scaler is None:
            loss.backward()

            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
            # weight_optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


        # logging
        torch.cuda.synchronize()
        if now_it % 100 == 0:
            print("[{}/{}]\t[{}/{}]\ttotal_loss:{}\tloss_ssl:{}\tloss_self:{}\t{}\t{}\tlr:{}\tw:{}".format(epoch, args.epochs, now_it, len(data_loader), round(loss.item(),4), round(loss_ssl.item(),4), round(self_loss_1.item(),4),\
                                                                                                   round(self_loss_2.item(),4), round(self_loss_3.item(),4), round(optimizer.param_groups[0]["lr"],6), round(optimizer.param_groups[0]["weight_decay"],4)))

        now_it = now_it + 1

    # gather the stats from all processes
    loss_epoch = {k: np.mean(meter) for k, meter in average_loss.items()}
    print("Averaged stats:", loss_epoch)
    return loss_epoch


class DeSDLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.register_buffer("loss_center", torch.zeros(1, 4))



        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        (feature_1, feature_2, feature_3, feature_4) = student_output
        feature_1 =  feature_1 / self.student_temp
        feature_2 =  feature_2 / self.student_temp
        feature_3 =  feature_3 / self.student_temp
        feature_4 =  feature_4 / self.student_temp

        student_out_4 = feature_4.chunk(self.ncrops)
        student_out_3 = feature_3.chunk(self.ncrops)
        student_out_2 = feature_2.chunk(self.ncrops)
        student_out_1 = feature_1.chunk(self.ncrops)



        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out_self_4 = F.softmax((teacher_output[-1] - self.center) / temp, dim=-1).detach().chunk(2)

        ssl_loss = 0
        self_loss_1 = 0
        self_loss_2 = 0
        self_loss_3 = 0
        n_loss_terms = 0
        for iq in  range(len(teacher_out_self_4)):
            for v in range(len(student_out_4)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss_1 = torch.sum(-teacher_out_self_4[iq] * F.log_softmax(student_out_1[v], dim=-1), dim=-1).mean()
                loss_2 = torch.sum(-teacher_out_self_4[iq] * F.log_softmax(student_out_2[v], dim=-1), dim=-1).mean()
                loss_3 = torch.sum(-teacher_out_self_4[iq] * F.log_softmax(student_out_3[v], dim=-1), dim=-1).mean()
                loss_4 = torch.sum(-teacher_out_self_4[iq] * F.log_softmax(student_out_4[v], dim=-1), dim=-1).mean()

                self_loss_1 +=  loss_1
                self_loss_2 += loss_2
                self_loss_3 += loss_3
                ssl_loss += loss_4
                n_loss_terms += 1

        ssl_loss /= n_loss_terms
        self_loss_1 /= n_loss_terms
        self_loss_2 /= n_loss_terms
        self_loss_3 /= n_loss_terms

        self.update_center(teacher_output[-1])

        total_loss = 0.25*ssl_loss + 0.25*self_loss_1 + 0.25*self_loss_2 + 0.25*self_loss_3

        return total_loss, ssl_loss, self_loss_1, self_loss_2, self_loss_3


    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeSD', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_DeSD(args)




