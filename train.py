import torch
import torch.nn as nn #神经网络模块
import torch.optim as optime #优化器
import torch.nn.functional as F #神经网络函数
import torch.utils #数据加载器
import torch.utils.data #数据加载器
from torchvision import models, transforms #计算机视觉模块

from perceptual import PerceptualLoss #导入自定义的感知损失函数

from PIL import Image #图像处理

from modle import ResNetAutoencoder #导入yzh创建的模型
from weld_autoencoder import AutoEncoder #导入hkl创建的模型

from tqdm import tqdm #进度条
from aim import Run #导入aim
from datetime import datetime #导入时间模块

import weld_dataset #导入数据加载器
import os #系统模块

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class Config:
    def __init__(self):
        self.data_list = 'list/weld1016_train_list.txt'
        self.batch_size = 16
        self.num_workers = 16
        self.shuffle = True
        self.epochs = 60
        self.learning_rate = 1e-3

def main(args):
    run = Run() #创建aim实例
    run["hparams"] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": args.shuffle,
        "epochs": args.epochs,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "learning_rate": args.learning_rate,
        'data_list': args.data_list
        
    }
    #加载数据
    train_loader = weld_dataset.train_loader(args.data_list, args.batch_size, args.num_workers, args.shuffle) #加载数据

    #初始化模型、损失函数、优化器
    # model = ResNetAutoencoder().cuda() #创建模型
    model = AutoEncoder().cuda() #创建模型
    criterion = PerceptualLoss().cuda() #创建损失函数
    optimizer = optime.Adam(model.parameters(), lr=args.learning_rate) #创建优化器

    scheduler = optime.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) #学习率调整器

    #模型参数
    require_grad_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))#获取需要梯度的参数
    all_parameters = list(model.parameters())#获取所有参数

    print("all_parameters:{}, require_grad_parameters:{}".format(len(all_parameters), len(require_grad_parameters)))#打印参数数量
    total_params = sum(p.numel() for p in model.parameters())#计算参数数量
    print('=> num of params: {} ({}M)'.format(total_params, int(total_params * 4 / (1024*1024))))#打印参数数量
    print(f"数据集：{len(train_loader)}")#打印数据集数量

    #训练模型
    for epoch in range(args.epochs):
        model.train()#设置为训练模式
        running_loss = 0.0 #初始化loss

        for images, _ in tqdm(train_loader):
            images = images.cuda()#将数据移到GPU
            # print('输入图像size:',images.shape)

            outputs = model(images)#前向传播

            recon_loss = F.mse_loss(outputs, images)#计算重建损失
            perceptual_loss_value = criterion(outputs, images)#计算感知损失
            total_loss = recon_loss + perceptual_loss_value#总损失


            optimizer.zero_grad()#梯度清零
            total_loss.backward()#反向传播
            optimizer.step()#更新参数

            running_loss += total_loss.item()

    
        # 在aim中保存训练得到的损失
        run.track(running_loss/len(train_loader), name='hkl autoencoder train loss', step=epoch, context={ "subset":args.data_list})
        
        if(epoch+1)%10==0:
            torch.save(model.state_dict(), f"results/weld_ResNetAutoencoder_{epoch}.pth")
            print(f"Model saved at epoch {epoch+1}")

        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

if __name__ == "__main__":
    args = Config()
    main(args)





