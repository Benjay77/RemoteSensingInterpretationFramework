"""
 *@Author: Benjay·Shaw
 *@CreateTime: 2022/7/12 14:49
 *@LastEditors: Benjay·Shaw
 *@LastEditTime:2022/7/12 14:49
 *@Description: 验证模块
"""
import torch
from tqdm import tqdm


def val(solver, mylog, model_path):
    # solver.net = torch.compile(solver.net)
    solver.net.eval()
    solver.net.load_state_dict(solver.load(model_path)['net'])
    with torch.no_grad():
        val_num = len(solver.val_loader)
        val_loader_iter = iter(solver.val_loader)

        loop = tqdm(val_loader_iter, total=val_num)
        val_epoch_loss = 0
        for img, mask in loop:
            solver.set_input(img, mask)
            solver.forward()
            predict = solver.net.forward(img)
            val_loss = solver.loss(solver.mask, predict)
            val_epoch_loss += val_loss.item()
            loop.set_postfix(loss=val_loss.item())

        val_epoch_loss /= val_num

        return val_epoch_loss
