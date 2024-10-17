import argparse
import pprint


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging', type=str, default='kd')
    parser.add_argument('--vit_path', type=str, default='/vit-large-patch16-224-in21k')
    parser.add_argument('--train_dir', type=str, default="i9/i9_original/train_watermark_m")
    parser.add_argument('--logit_path', type=str, default="./results/90_train_output.csv")
    parser.add_argument('--att_path', type=str, default="./attention/sorted_attention_i9_output.txt")
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--num_classes', type=int, default=9)
    parser.add_argument('--lr', type=float, default=1e-4)#0.0001
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--brier_weight', type=float, default=0.1, help='brier_weight')
    
    parser.add_argument('--mask_portion', type=float, default=0.1, help='mask_portion')
    parser.add_argument('--plat_percent', type=float, default=0.1, help='platt_percent')
    parser.add_argument('--temperature', type=float, default=2.0, help='temperature')
    parser.add_argument('--kd_weight', type=float, default=0.01, help='ke_weight')
    parser.add_argument('--exp_weight', type=float, default=0.001, help='exp_weight')
    parser.add_argument('--epoch_num', type=int, default=90) 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--info_dim', type=int, default=1024)
    parser.add_argument('--milestones',
                        type=list,
                        default=[30, 60, 90, 120],
                        help='optimizer milestones')
    parser.add_argument('--warmup_percent', type=float, default=0.1)  #c
    
    parser.add_argument('--gpu', type=int, default=0, help='which gpus to use')
    args = parser.parse_args()
    
    
    # pprint.PrettyPrinter().pprint(args.__dict__)
    return args