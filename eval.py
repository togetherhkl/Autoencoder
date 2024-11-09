from weld_autoencoder import AutoEncoder
from modle import ResNetAutoencoder
from perceptual import PerceptualLoss
import weld_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from aim import Run
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def eval():
    data_list = 'list/weld1016_test_list.txt'
    batch_size = 1
    num_workers = 16
    shuffle = False

    run = Run()
    run["hparams"] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "shuffle": shuffle,
        "data_list": data_list
    }
    run["description"] = "Evaluate the model on the test set."


    val_loader = weld_dataset.test_loader(data_list, batch_size, num_workers, shuffle)
    model = AutoEncoder().cuda()
    state_dict=torch.load('results/weld_ResNetAutoencoder_59.pth')

    model.load_state_dict(state_dict, strict=False)

    
    criterion = PerceptualLoss().cuda()

    output_dir = 'reconstructed_images'
    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    plt.figure(figsize=(16, 9))
    eval_loss = 0.0
    

    with torch.no_grad():
        # progress_bar = tqdm(enumerate(val_loader), desc='Evaluating', unit='batch', total=len(val_loader))
        i = 1
        for images, _ in tqdm(val_loader):
            # if i == 63:
            #     break
            images = images.cuda()
            outputs = model(images) 

            recon_loss = nn.functional.mse_loss(outputs, images)
            perceptual_loss = criterion(outputs, images)
            total_loss = recon_loss + perceptual_loss

            run.track(total_loss/len(val_loader), name='hkl autoencoder test loss', context={"subset": data_list})

            # eval_loss += total_loss.item()
            # print("eval_loss:",eval_loss)


            # loss = criterion(outputs, images)
            images = transforms.ToPILImage()(images.squeeze().cpu())
            outputs = transforms.ToPILImage()(outputs.squeeze().cpu())

            
            '''
            if i % 63 != 0:#每隔63
                tmp = i % 63
                plt.subplot(8,16,2*tmp+1, xticks=[], yticks=[])
                plt.imshow(images)

                plt.subplot(8,16,2*tmp+2, xticks=[], yticks=[])
                plt.imshow(outputs)
                i += 1
            else:
                plt.savefig(output_dir + '/weld1016_autoencoder60_'+str(i)+'.png')
                #清空画布
                plt.clf()
                plt.figure(figsize=(16, 9))
                i += 1
            '''

  
            '''
            # progress_bar.set_postfix(loss=loss.item())
            for j in range(images.size(0)):
            # 保存原图
                original_image = images[j].cpu().numpy().transpose(1, 2, 0)
                original_image = (original_image * 255).astype(np.uint8)
                Image.fromarray(original_image).save(os.path.join(output_dir, f'100_original_{i*images.size(0)+j}.png'))
                
                # 保存重建图
                output_image = outputs[j].cpu().numpy().transpose(1, 2, 0)
                output_image = (output_image * 255).astype(np.uint8)
                Image.fromarray(output_image).save(os.path.join(output_dir, f'100_reconstructed_{i*images.size(0)+j}.png'))
            '''
    
    # run.track(eval_loss/len(val_loader), name='hkl autoencoder test loss', context={"subset": data_list})
    # plt.savefig(output_dir + '/weld1016_autoencoder60.png')
  

if __name__ == '__main__':
    eval()