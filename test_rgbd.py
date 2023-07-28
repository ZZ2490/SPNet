import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from Encoder_Prototype_rgbd_ablation import Mnet
from data import test_dataset
import time
from thop import profile

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path', type=str, default='../rgb_d/test_d/', help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

#load the model
model = Mnet()

model.cuda()
model.eval()
fps = 0

# RGBT-Test
if __name__ == '__main__':
    for i in ['MAE',]:   #,'AVG',
        test_datasets = ['DES','NJU2K','STERE','NLPR','SIP']
        for dataset in test_datasets:
            time_s = time.time()
            model_path = os.path.join('./model/rgb_d/', '3096_SPNet_gwm_tha_Best_' + str(i) + '_epoch_20p.pth') #Best_AVG_  ，  epoch_180   best_MAE_epoch
            model.load_state_dict(torch.load(model_path))
            sal_save_path = os.path.join('./output/', dataset + '-' + str(i) + '-3096_SPNet_gwm_tha_Best_MAE_20p/')
            if not os.path.exists(sal_save_path):
                os.makedirs(sal_save_path)
                # os.makedirs(edge_save_path)
            image_root = dataset_path + dataset +'/RGB/'
            gt_root = dataset_path + dataset + '/GT/'
            t_root = dataset_path + dataset + '/depth/'
            test_loader = test_dataset(image_root, gt_root, t_root, opt.testsize)
            nums = test_loader.size
            # r_input = torch.randn(1, 3, 352, 352).cuda().float()
            # t_input = torch.randn(1, 3, 352, 352).cuda().float()
            # flops, parameters = profile(model, (t_input, t_input,))
            # print("The number of flops: {}".format(flops))
            # print("The number of params: {}".format(parameters))

            # map_glo = torch.unsqueeze(torch.mean(out2, 1), 1)
            # map_glo = (map_glo - map_glo.min()) / (map_glo.max() - map_glo.min() + 1e-8)
            # map_glo = F.interpolate(map_glo, size=gt.shape, mode='bilinear', align_corners=True)
            # map_glo = np.squeeze(torch.Tensor(map_glo.cpu().data.numpy()).cpu().data.numpy())
            # heatmap = cv2.applyColorMap(np.uint8(255 * map_glo), cv2.COLORMAP_JET)
            # cv2.imwrite(os.path.join('./feture map/', 'main_in_sup_2.png'), heatmap)
            w_dict = {}
            for j in range(test_loader.size):
                image, gt, t, name, image_for_post = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                t = t.cuda()
                score, score1, score2, s_sig = model(image, t)

                res = F.upsample(score2, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                print('save img to: ', sal_save_path + name[:-4] + '.png')
                cv2.imwrite(os.path.join(sal_save_path, name[:-4] + '.png'), res * 255)
            time_e = time.time()
            fps += (nums / (time_e - time_s))
            print("FPS:%f" % (nums / (time_e - time_s)))
            print('Test Done!')
        print("Total FPS %f" % fps) # this result include I/O cost

