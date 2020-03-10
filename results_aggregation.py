import os
import argparse
import json
import pandas as pd
options = ["h_rank",
           "l_rank",
           "HL",
           "l_rank_identity",
           "HL_identity",
           "dual_low_z_v",
           "dual_low_zi_v",
           "dual_low_z_vi",
           "dual_low_zi_vi"]

metrics = ["loss", "accuracy", "val_loss","val_accuracy"]

def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep CNN.")
    parser.add_argument('--option', nargs='?', default='h_rank',
                        help='Option for Conv weight matrix.')
    parser.add_argument('--history_path', nargs='?', default='./history/',
                        help='The folder path to save history.')
    parser.add_argument('--target_folder', nargs='?', default='./aggregated_results',
                        help='Option for Conv weight matrix.')
    parser.add_argument('--target_rank', type=int, default=2,
                        help='Target rank for Conv low rank weight matrix.')
    parser.add_argument('--dataset', nargs='?', default='mnist',
                        help='Choose a Dataset')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Training Epoch')
    parser.add_argument('--channel', type=int, default=1,
                        help='Number of channel.')
    parser.add_argument('--kernel', type=int, default=1,
                        help='Number of kernel.')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='The dimension for a kernel.')
    return parser.parse_args()

def load_history(args):
    model_str = "_".join([args.option, "kernel_size", str(args.kernel_size),
                          "num_kernel", str(args.kernel),
                          "channel", str(args.channel),
                          "rank", str(args.target_rank),
                          "epoch", str(args.epoch)])
    target_file = os.path.join(args.history_path, args.dataset, model_str+".csv")
    print(target_file)
    df_history = pd.read_csv(target_file)
    df_history = pd.read_json(target_file)
    if not os.path.isdir(args.target_folder):
        os.makedirs(args.target_folder)
    df_history.to_csv(os.path.join(args.target_folder,'history_'+str(args.dataset)+'_'+str(args.option)+'.csv'),index=False)




if __name__ == '__main__':
    # Data loading
    args = parse_args()
    load_history(args)

