"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/7/12 14:49
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/7/12 14:49
 *@Description: 训练模块
"""
import argparse
import math
import os
import time
import warnings

import torch
from tqdm import tqdm

from utils.cawb import CosineAnnealingWarmbootingLR
from utils.my_net import MyNet
from utils.plot import argparses_plot
from val import val


def getargs():
    parse = argparse.ArgumentParser()
    parse.add_argument('--image_size', type=int, default=512)
    parse.add_argument('--resolution', type=str, default='0.3m')
    parse.add_argument('--interpret_type', type=str, default='build')
    parse.add_argument('--weight_name', type=str, default=time.strftime("%Y%m%d%H%M", time.localtime()))
    parse.add_argument('--image_type', type=str, default='.tif')
    parse.add_argument('--label_type', type=str, default='.tif')
    parse.add_argument('--lr_init', type=float, default=0.03)
    parse.add_argument('--lrf', type=float, default=0.2)
    parse.add_argument('--kronecker_r1', nargs='+', type=int,
                       default=[2, 4, 8])
    parse.add_argument('--kronecker_r2', nargs='+', type=int,
                       default=[1, 3, 5])
    parse.add_argument('--total_epoch', type=int, default=500)
    parse.add_argument('--arch', '-a', metavar='ARCH', default='UNetR50',
                       help='UNetR50')
    parse.add_argument('--loss_function', type=str, default='BceDiceLoss',
                       help='BceDiceLoss')
    parse.add_argument('--batchsize_per_card', type=int, default=96)
    parse.add_argument('--use_multiple_GPU', type=bool, default=False
                       )
    parse.add_argument('--device', default='0,1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parse.add_argument('--use_cosine_lr', type=bool, default=False)
    parse.add_argument('--dataset_dir', default='./dataset/',
                       help='dataset dir')
    parse.add_argument('--weights', type=str, default='./inference/weights', help='path of weights files')
    parse.add_argument('--resume', type=bool, default=False)
    parse.add_argument('--resume_weight_name', type=str, default='202110062111')
    parse.add_argument('--log_dir', default='./logs', help="log dir")
    parse.add_argument('--train_epoch_best_loss', type=float, default=100.)
    parse.add_argument('--val_epoch_best_loss', type=float, default=100.)
    parse.add_argument('--best_iou', type=float, default=0.)
    parse.add_argument('--cawb_steps', nargs='+', type=int,
                       default=[25, 55, 85, 115, 145, 175, 205, 235, 265, 295, 325, 355, 385, 415, 445, 475])
    parse.add_argument('--update_lr_epoch', type=int, default=6)
    parse.add_argument('--early_stop_epoch', type=int, default=10)
    parse.add_argument('--num_workers', type=int, default=4)
    return parse.parse_args()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # parameters
    args = getargs()

    # model
    solver = MyNet(args)

    # log
    log_path = os.path.join(args.log_dir, args.resolution, args.interpret_type)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    my_log = open(os.path.join(log_path, args.interpret_type + args.weight_name + '.txt'), 'w')

    # epoch
    if not os.path.exists(os.path.join(args.weights, args.resolution, args.interpret_type,
                                       args.interpret_type + args.weight_name)):
        os.makedirs(os.path.join(args.weights, args.resolution, args.interpret_type,
                                 args.interpret_type + args.weight_name))
    tic = time.time()
    begin_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    my_log.write(str('training begin time: ') + begin_time + '\n')
    print('training begin time: ', begin_time)
    training_info = str(
        'training loss_function: ') + args.loss_function + ';' + str(
        'training dataset: ') + args.dataset_dir.split('/dataset/')[
                        -1] + ';' + str('training image_size: ') + str(args.image_size) + ';' + str(
        'training lr_init: ') + str(args.lr_init) + ';' + str('training lrf: ') + str(args.lrf) + ';' + str(
        'training batch_size: ') + str(solver.batch_size) + ';' + str(
        'training use_cosine_lr: ') + str(
        args.use_cosine_lr) + ';' + str(
        'training kronecker_r1: ') + str(args.kronecker_r1) + ';' + str(
        'training kronecker_r2: ') + str(args.kronecker_r2) + ';' + str('training model: ') + str(solver.net) + '\n'
    if args.resume:
        training_info = str('resume:') + str(args.resume) + ';' + str(
            'resume_weight_name:') + args.resume_weight_name + ';' + training_info
    my_log.write(training_info)
    print(training_info)

    no_optim = 0
    start_epoch = 0
    end_epoch = 0
    old_lr = new_lr = args.lr_init
    train_loss_list = []
    val_loss_list = []

    model_path = os.path.join(args.weights, args.resolution, args.interpret_type,
                              args.interpret_type + args.weight_name,
                              args.interpret_type + args.weight_name)
    if args.resume:
        resume_model_path = os.path.join(args.weights, args.resolution, args.interpret_type,
                                         args.interpret_type + args.resume_weight_name,
                                         args.interpret_type + args.resume_weight_name)
        resume_model = solver.load(resume_model_path + '_last.pth')
        solver.net.load_state_dict(resume_model['net'])
        solver.optimizer.load_state_dict(resume_model['optimizer'])
        start_epoch = resume_model['cur_epoch']
        train_loss_list = resume_model['train_loss_list']
        val_loss_list = resume_model['val_loss_list']
        if len(train_loss_list) > len(val_loss_list):
            del train_loss_list[-1]
            start_epoch = start_epoch - 1

    for epoch in range(start_epoch + 1, args.total_epoch + 1):
        if epoch == int(args.total_epoch * args.lrf) and args.use_cosine_lr:
            solver.old_lr = 0.01
            solver.optimizer = torch.optim.SGD(params=solver.net.parameters(), lr=solver.old_lr, momentum=0.9,
                                               weight_decay=1e-5, nesterov=True)
            lf = lambda x, y=args.total_epoch - epoch: (((1 + math.cos(x * math.pi / y)) / 2) ** 1.0) * (
                    1 - args.lrf) + args.lrf
            solver.scheduler = CosineAnnealingWarmbootingLR(solver.optimizer, epochs=args.total_epoch - epoch,
                                                            steps=args.cawb_steps, step_scale=0.7,
                                                            lf=lf, batchs=solver.batch_size, warmup_epoch=3,
                                                            epoch_scale=4.0)
        train_loader_iter = iter(solver.train_loader)
        loop = tqdm(train_loader_iter, total=len(solver.train_loader))
        train_epoch_loss = 0
        val_epoch_loss = 0
        cuda_epoch_mem = 0
        for img, mask in loop:
            solver.set_input(img, mask)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
            cuda_mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
            cuda_epoch_mem += cuda_mem
            loop.set_description(f'Epoch[{epoch}/{args.total_epoch}]  CUDA {cuda_mem:.3g}GB')
            loop.set_postfix(loss=train_loss)

        # Scheduler and log lr
        old_lr = solver.old_lr
        if args.use_cosine_lr:
            solver.scheduler.step()
            for param_group in solver.optimizer.param_groups:
                new_lr = param_group['lr']
                solver.old_lr = param_group['lr']

        # train_epoch_loss
        train_epoch_loss /= len(solver.train_loader)
        train_loss_list.append(train_epoch_loss)
        save_model = {'net': solver.net.state_dict(), 'optimizer': solver.optimizer.state_dict(),
                      'cur_epoch': epoch, 'train_loss_list': train_loss_list, 'val_loss_list': val_loss_list}
        solver.save(model_path + '_last.pth', save_model)

        # val
        val_epoch_loss = val(solver, my_log, model_path + '_last.pth')
        val_loss_list.append(val_epoch_loss)
        save_model['val_loss_list'] = val_loss_list
        solver.save(model_path + '_last.pth', save_model)
        solver.net.train()

        # CUDA Memory
        cuda_epoch_mem /= len(solver.train_loader)

        end_epoch = epoch

        if train_epoch_loss < args.train_epoch_best_loss and val_epoch_loss < args.val_epoch_best_loss:
            no_optim = 0
            args.train_epoch_best_loss = train_epoch_loss
            args.val_epoch_best_loss = val_epoch_loss
            solver.save(model_path + '_best.pth', None, True)
        else:
            no_optim += 1
            # update learning-rate
            if no_optim > args.update_lr_epoch and not args.use_cosine_lr:
                if old_lr < 5e-7:
                    break
                # solver.save(os.path.join(args.weights, args.weight_name, args.weight_name + '_last.pth'))
                new_lr = solver.update_lr(args.lrf, factor=True, my_log=my_log)

        my_log.write('********************' + '\n')
        log = ('--epoch: ' + str(epoch) + '  --time: ' + str(
            int(time.time() - tic)) + '  --cuda_epoch_memory: ' + str(f'{cuda_epoch_mem:.3g}GB')
               + '  --update learning rate: ' + str(
            old_lr) + ' -> ' + str(
            new_lr) + '  --no_optim: ' + str(no_optim) + '  --train_epoch_best_loss: ' + str(
            args.train_epoch_best_loss) + '  --val_epoch_best_loss: ' + str(
            args.val_epoch_best_loss) + '  --train_epoch_loss: ' + str(
            train_epoch_loss) + '  --val_epoch_loss: ' + str(
            val_epoch_loss) + '\n')
        my_log.write(log)
        print(log)

        # EarlyStopping
        if no_optim > args.early_stop_epoch:
            my_log.write('early stop at %d epoch' % epoch + '\n')
            print('early stop at %d epoch' % epoch + '\n')
            break
        my_log.flush()
    argparses_plot(args, 'train_loss&val_loss', end_epoch, train_loss_list, val_loss_list)
    time2 = time.localtime()
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time2)
    print(f'\ntraining end_time: ', end_time)
    my_log.write(str('training end time: ') + end_time + '\n')
    my_log.write(f'{end_epoch} epochs completed in {(time.time() - tic) / 3600:.3f} hours.' + '\n')
    print(f'\n{end_epoch} epochs completed in {(time.time() - tic) / 3600:.3f} hours.')
    my_log.write('Train Finish' + '\n')
    print(f'\nTrain Finish!')
    torch.cuda.empty_cache()
    my_log.close()
