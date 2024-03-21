"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/7/12 14:49
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/7/12 14:49
 *@Description: 网络模型
"""
import math
import os

from torch.autograd import Variable as V

from networks.unet_r50 import UNetR50
from utils.cawb import CosineAnnealingWarmbootingLR
from utils.data_process import ImageFolder
from utils.loss import *
from lightning import Fabric


class MyNet:
    def __init__(self, args, evalmode=False):
        # lightning
        # torch.set_float32_matmul_precision('high')
        fabric = Fabric(accelerator="cuda", precision="bf16-mixed")
        fabric.launch()
        self.img = None
        self.mask = None
        self.img_id = None
        self.fabric = fabric
        if args.arch == 'UNetR50':
            self.net = UNetR50()

        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_type = device.type
        if device.type == 'cuda':
            if args.use_multiple_GPU:
                device_ids = range(torch.cuda.device_count())
            else:
                device_ids = [0]
            self.net = nn.DataParallel(self.net.cuda(device), device_ids=device_ids)
            self.batch_size = torch.cuda.device_count() * args.batchsize_per_card
        else:
            self.net = nn.DataParallel(self.net)
            self.batch_size = args.batchsize_per_card
        # if args.use_multiple_GPU:
        #     self.net = nn.DataParallel(self.net.cuda(), device_ids=range(torch.cuda.device_count()))
        #     self.batch_size = torch.cuda.device_count() * args.batchsize_per_card
        # else:
        #     os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        #     if torch.cuda.is_available():
        #         self.net = nn.DataParallel(self.net, device_ids=[0])
        #     self.batch_size = args.batchsize_per_card
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=args.lr_init)
        if args.use_cosine_lr:
            lf = lambda x, y=args.total_epoch: (((1 + math.cos(x * math.pi / y)) / 2) ** 1.0) * (
                    1 - args.lrf) + args.lrf
            self.scheduler = CosineAnnealingWarmbootingLR(self.optimizer, epochs=args.total_epoch,
                                                          steps=args.cawb_steps, step_scale=0.7, lf=lf,
                                                          batchs=self.batch_size,
                                                          warmup_epoch=3, epoch_scale=4.0)

        if args.loss_function == 'BceDiceLoss':
            self.loss = BceDiceLoss(self.net)
        self.old_lr = args.lr_init

        # dataset
        type_length = len(args.image_type)
        root_train = args.dataset_dir + '/train/'  # codetest
        # train_list = list(map(lambda x: x[:-8], filter(lambda x: x.find('_mask') != -1, os.listdir(root_train))))
        train_list = list(map(lambda x: x[:-type_length], os.listdir(root_train + 'imgs')))
        dataset_train = ImageFolder(train_list, root_train, args.image_type, args.label_type)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers, pin_memory=True)

        root_val = args.dataset_dir + '/val/'
        # val_list = list(map(lambda x: x[:-8], filter(lambda x: x.find('_mask') != -1, os.listdir(root_val))))
        val_list = list(map(lambda x: x[:-type_length], os.listdir(root_val + 'imgs')))
        dataset_val = ImageFolder(val_list, root_val, args.image_type, args.label_type)
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=args.num_workers, pin_memory=True)
        # self.train_loader, self.val_loader = train_loader, val_loader
        if evalmode:
            # for i in self.net.modules():
            #     if isinstance(i, nn.BatchNorm2d):
            #         i.eval()
            self.net.eval()
        else:
            self.net, self.optimizer = self.fabric.setup(self.net, self.optimizer)
            self.train_loader, self.val_loader = fabric.setup_dataloaders(train_loader, val_loader)
            # self.net = torch.compile(self.net)
            self.net.train()

    def set_input(self, img_batch, mask_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.img_id = img_id

    def forward(self, volatile=False):
        if self.device_type == 'cuda':
            self.img = V(self.img.cuda(), volatile=volatile)
        else:
            self.img = V(self.img, volatile=volatile)
        if self.mask is not None:
            if self.device_type == 'cuda':
                self.mask = V(self.mask.cuda(), volatile=volatile)
            else:
                self.mask = V(self.mask, volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad(set_to_none=True)
        pred = self.net.forward(self.img)
        loss = self.loss(self.mask, pred)
        self.fabric.backward(loss)
        # loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path, save_model, is_best=False):
        if is_best:
            torch.save(self.net.state_dict(), path, _use_new_zipfile_serialization=False)
        else:
            torch.save(save_model, path, _use_new_zipfile_serialization=False)

    def load(self, path, is_best=False):
        if is_best:
            self.net.load_state_dict(torch.load(path))
        else:
            return torch.load(path)

    def update_lr(self, lrf, my_log, factor=False):
        if factor:
            new_lr = self.old_lr * lrf
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        my_log.write('update learning rate: %f -> %f' % (self.old_lr, new_lr) + '\n')
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr) + '\n')
        self.old_lr = new_lr
        return new_lr
