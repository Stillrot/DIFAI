import argparse
from torch.backends import cudnn
import torch
from mlgn import MLGN
from difai import DIFAI

def main(args):
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    if 'mlgn' in args.mode:
        mlgn_ = MLGN(args)
    else:
        difai_ = DIFAI(args)

    if args.mode == 'mlgn_train':
        mlgn_.train(args)

    elif args.mode == 'mlgn_val':
        mlgn_.val(args)

    elif args.mode == 'mlgn_test':
        mlgn_.test(args)

    elif args.mode == 'train':
        difai_.train(args)

    elif args.mode == 'val':
        difai_.val(args)

    elif args.mode == 'test':
        difai_.test(args)

    else:
        raise NotImplementedError

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=7777, help='random seed')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'val', 'test', 'mlgn_train', 'mlgn_val', 'mlgn_test'], help='This argument is used in solver')

    parser.add_argument('--z_dim', type=int, default=256)
    parser.add_argument('--start_h', type=int, default=8)
    parser.add_argument('--start_w', type=int, default=8)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--mlgn_lambda_gp', type=float, default=10, help='gp loss')

    parser.add_argument('--lambda_re_vgg', type=float, default=10)
    parser.add_argument('--lambda_fm', type=float, default=10)
    parser.add_argument('--lambda_pdiv', type=float, default=1)

    parser.add_argument('--lambda_adv', type=float, default=0.5)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--lambda_hole', type=float, default=3)
    parser.add_argument('--lambda_valid', type=float, default=0.5)
    parser.add_argument('--lambda_ssim', type=float, default=3)
    parser.add_argument('--lambda_prc', type=float, default=1)
    parser.add_argument('--lambda_style', type=float, default=120)

    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_step', type=int, default=2500)
    parser.add_argument('--verbose_step', type=int, default=50)

    parser.add_argument('--beta1', type=float, default=0.0, help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99, help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--d_lr', type=float, default=0.0004, help='Learning rate')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--total_iters', type=int, default=400000, help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0, help='Iterations to resume training/testing')

    parser.add_argument('--mlgn_val_dir', type=str, default='./mlgn_validate', help='validation MLGN sample save directory')
    parser.add_argument('--mlgn_checkpoint_dir', type=str, default='mlgn_ckpt', help='MLGN checkpoint files directory')
    parser.add_argument('--pretrained_mlgn_step', type=int, default=100000)

    parser.add_argument('--image_dir', type=str, default='../data/train/CelebA_HQ', help='image directory')
    parser.add_argument('--image_list_dir', type=str, default='../data/train/CelebAMask-HQ-attribute-anno.txt', help='label directory')
    parser.add_argument('--masks_dir', type=str, default='../data/masks/train', help='masks directory')

    parser.add_argument('--image_val_dir', type=str, default='../data/val/CelebA_HQ', help='validation image directory')
    parser.add_argument('--masks_val_dir', type=str, default='../data/masks/test', help='validation masks directory')
    parser.add_argument('--val_sample_dir', type=str, default='val_sample', help='validation samples results files directory')

    parser.add_argument('--image_test_dir', type=str, default='user_test/input/image', help='test images files directory')
    parser.add_argument('--masks_test_dir', type=str, default='user_test/input/mask', help='test mask files directory')
    parser.add_argument('--test_sample_dir', type=str, default='user_test', help='test samples results files directory')
    parser.add_argument('--checkpoint_dir', type=str, default='ckpt', help='checkpoint files directory')

    #PSP ENCODER
    parser.add_argument('--psp_checkpoint_path', default='./models/pSp/psp_ckpt/psp_ffhq_encode.pt', type=str, help='Path to pSp model checkpoint')
    parser.add_argument('--resize_outputs', type=bool, default=True, help='Whether to resize outputs to 256x256 or keep at 1024x1024')
    #STYLEGAN2
    parser.add_argument('--stylegan2_checkpoint_path', default='./models/styleGAN2/ckpt/stylegan2_ffhq1024.pth', type=str, help='Path to StyleGAN2 model checkpoint')
    parser.add_argument('--layer_idx', default='all', type=str)#,  choices=['all', '0-1', '2-5', '6-13'])
    parser.add_argument('--semantic_layer', default=0, type=int)
    parser.add_argument('--style_sample_num', default=5, type=int)
    parser.add_argument('--start_distance', default=-3, type=int)
    parser.add_argument('--end_distance', default=3, type=int)

    args = parser.parse_args()

    main(args)