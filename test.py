import os
import numpy as np
import torch
from os import listdir
import torch.nn as nn
from sewar import uqi
from torch.autograd import Variable
from option import opt
from data_utils import is_image_file
from modelbest1_x2 import TCHISRNet
import scipy.io as scio
from evaluate import PSNR, SSIM, SAM,RMSE,ERGAS



def main():
    input_path = '/home/yanglz/Test_TCHISRNet/Data/dataCAVE/valid/CAVE/2/'
    out_path = '/home/yanglz/Test_TCHISRNet/Data/dataCAVE/result/ablation/nolyT/epoch27/'

    # input_path = '/home/yanglz/Test_TCHISRNet/Data/dataHarvard/valid/Hararvd/4/'
    # out_path = '/home/yanglz/Test_TCHISRNet/Data/dataHarvard/result/Hararvd/4/epoch44/'

    #input_path = '/home/yanglz/Test_TCHISRNet/Data/datachikusei/valid/chikusei/2/'
    #out_path = '/home/yanglz/Test_TCHISRNet/Data/datachikusei/result/chikusei/2/epoch8/'



    if not os.path.exists(out_path):
        os.makedirs(out_path)



    PSNRs = []
    SSIMs = []
    SAMs = []
    RMSEs=[]
    ERGASs=[]
    UIQIs=[]#Universal  Image  Quality  Index


    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if opt.cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    model = TCHISRNet(opt)


    model = nn.DataParallel(model).cuda()

    checkpoint = torch.load('/home/yanglz/Test_TCHISRNet/Coda/TCHISRNet-master/checkpoint_CAVEonlyT_X2/model_2_epoch_27.pth')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    images_name = [x for x in listdir(input_path) if is_image_file(x)]

    for index in range(len(images_name)):
      with torch.no_grad():
        mat = scio.loadmat(input_path + images_name[index])
        hyperLR = mat['LR'].astype(np.float32).transpose(2, 0, 1)
        HR = mat['HR'].astype(np.float32).transpose(2, 0, 1)

        input = Variable(torch.from_numpy(hyperLR).float()).contiguous().view(1, -1, hyperLR.shape[1],
                                                                                             hyperLR.shape[2])
        SR = np.array(HR).astype(np.float32)


        input = input.cuda()

        localFeats = []

        for i in range(input.shape[1]):

            if i == 0:
                x = input[:, 0:3, :, :]
                y = input[:, 0, :, :]

            elif i == input.shape[1] - 1:
                x = input[:, i - 2:i + 1, :, :]
                y = input[:, i, :, :]
            else:
                x = input[:, i - 1:i + 2, :, :]
                y = input[:, i, :, :]

            output, localFeats = model(x, y, localFeats, i)
            SR[i, :, :] = output.cpu().data[0].numpy()



        SR[SR < 0] = 0
        SR[SR > 1.] = 1.

        psnr = PSNR(SR, HR)
        ssim = SSIM(SR, HR)
        sam = SAM(SR, HR)
        rmse = RMSE(SR, HR)
        ergas = ERGAS(SR, HR)
        UIQI = uqi(SR, HR, ws=8)

        PSNRs.append(psnr)
        SSIMs.append(ssim)
        SAMs.append(sam)
        RMSEs.append(rmse)
        ERGASs.append(ergas)
        UIQIs.append(UIQI)


        SR = SR.transpose(1, 2, 0)
        HR = HR.transpose(1, 2, 0)

        scio.savemat(out_path + images_name[index], {'HR': HR, 'SR': SR})
        print(
            "===The {}-th picture=====PSNR:{:.3f}=====SSIM:{:.4f}=====SAM:{:.3f}=======RMSE:{:.3f}=====ERGAS:{:.3f}=====UIQI:{:.3f}====Name:{}".format(
                index + 1, psnr, ssim, sam, rmse, ergas, UIQI, images_name[index], ))
    print(
          "=====averPSNR:{:.3f}=====averSSIM:{:.4f}=====averSAM:{:.3f}=====averRMSE:{:.3f}=====averERGAS:{:.3f}=====averUIQI:{:.3f}".format(
              np.mean(PSNRs), np.mean(SSIMs), np.mean(SAMs), np.mean(RMSEs), np.mean(ERGASs), np.mean(UIQIs)))


if __name__ == "__main__":
    main()