import os
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
from torch import nn
from datetime import datetime
from Encoder_Prototype_rgbd import Mnet
import torch.optim as optim
from torchvision.utils import make_grid
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from torch.nn import functional as F
from smooth_loss import get_saliency_smoothness
import torch.backends.cudnn as cudnn
from option_d import opt
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def my_loss1(score, score1, score2, s_sig,  label):

    sal_loss1 = structure_loss(score1, label)
    sal_loss2 = structure_loss(score2, label)
    sal_loss = structure_loss(score, label)
    sml = get_saliency_smoothness(s_sig, label)

    return sal_loss + 0.8 * sal_loss1 + 0.8 * sal_loss2 + 0.5 * sml


# set the device for training
cudnn.benchmark = True

train_image_root = opt.train_rgb_root
train_gt_root = opt.train_gt_root
train_t_root = opt.train_t_root

val_image_root = opt.val_rgb_root
val_gt_root = opt.val_gt_root
val_t_root = opt.val_t_root

save_path = opt.save_path

model = Mnet()
num_parms = 0
model.cuda()
for p in model.parameters():
    num_parms += p.numel()
print("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path
if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(train_image_root, train_gt_root, train_t_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader  = test_dataset(val_image_root, val_gt_root, val_t_root, testsize = 384)
total_step = len(train_loader)

# set loss function
step = 0
best_mae   = 1
best_mae_epoch = 0
best_avg_loss = 1
best_avg_epoch = 0
# writer = SummaryWriter(save_path + 'summary')

# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step, best_avg_loss, best_avg_epoch
    model.train()
    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts, t) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            rgb = images.cuda()
            gts = gts.cuda()
            t = t.cuda()
            score, score1, score2, s_sig = model(rgb, t)
            sal_loss = my_loss1(score, score1, score2, s_sig, gts)
            loss = sal_loss
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 30 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||sal_loss:{:4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             optimizer.state_dict()['param_groups'][0]['lr'], loss.data))

        loss_all /= epoch_step
        print('Epoch [{:03d}/{:03d}] || epoch_avg_loss:{:4f} ... best avg loss: {} ... best avg epoch: {}'.format(epoch, opt.epoch, loss_all.data, best_avg_loss, best_avg_epoch))
        if epoch == 1:
            best_avg_loss = loss_all
        else:
            if best_avg_loss >= loss_all:
                best_avg_loss = loss_all
                best_avg_epoch = epoch
                torch.save(model.state_dict(), save_path + '3090_test_SPNet_Best_AVG_epoch_20p.pth')
                print('best avg epoch:{}'.format(epoch))
        if epoch == 200:
            torch.save(model.state_dict(), save_path + 'epoch_{}_3090_test_SPNet_20p.pth'.format(epoch))

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # torch.save(model.state_dict(), save_path + 'epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise

def val(test_loader,model,epoch,save_path):
    global best_mae,best_mae_epoch
    model.eval()
    with torch.no_grad():
        mae_sum=0
        for i in range(test_loader.size):
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt      = np.asarray(gt, np.float32)
            gt     /= (gt.max() + 1e-8)
            image   = image.cuda()
            depth   = depth.cuda()
            score, score1, score2, s_sig = model(image,depth)
            res     = F.interpolate(score, size=gt.shape, mode='bilinear', align_corners=False)
            res     = res.sigmoid().data.cpu().numpy().squeeze()
            res     = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res-gt))*1.0/(gt.shape[0]*gt.shape[1])

        mae = mae_sum/test_loader.size
        print('SPNet_Epoch: {} -MAE: {}  ...  bestmae: {} ...  best mae Epoch: {}'.format(epoch,mae,best_mae,best_mae_epoch))
        if epoch==1:
            best_mae = mae
        else:
            if mae<best_mae:
                best_mae   = mae
                best_mae_epoch = epoch
                torch.save(model.state_dict(), save_path+'3090_test_SPNet_Best_MAE_epoch_20p.pth')
                print('SPNet mae epoch:{}'.format(epoch))

if __name__ == '__main__':
    print("SPNet_Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        # writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        if epoch % 2 ==0:
            val(test_loader, model, epoch, save_path)