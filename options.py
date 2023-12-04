import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=10, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=60, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default='', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

# RGB-T Datasets
parser.add_argument('--train_rgb_root', type=str, default='../RGBT_dataset/train/RGB/', help='the training RGB images root')
parser.add_argument('--train_t_root', type=str, default='../RGBT_dataset/train/T/', help='the training Thermal images root')
parser.add_argument('--train_gt_root', type=str, default='../RGBT_dataset/train/GT/', help='the training GT images root')

parser.add_argument('--val_rgb_root', type=str, default='../RGBT_dataset/train/RGB/', help='the training RGB images root')
parser.add_argument('--val_t_root', type=str, default='../RGBT_dataset/train/T/', help='the training Thermal images root')
parser.add_argument('--val_gt_root', type=str, default='../RGBT_dataset/train/GT/', help='the training GT images root')

parser.add_argument('--save_path', type=str, default='./model/', help='the path to save models')
opt = parser.parse_args()