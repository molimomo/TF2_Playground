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

identity_options = ["l_rank_identity",
                    "HL_identity",
                    "dual_low_zi_v",
                    "dual_low_z_vi",
                    "dual_low_zi_vi"
                    ]

metrics = ["loss", "accuracy", "val_loss","val_accuracy"]

def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep CNN.")
    parser.add_argument('--option', nargs='?', default='h_rank',
                        help='Option for Conv weight matrix.')
    parser.add_argument('--history_path', nargs='?', default='./history/',
                        help='The folder path to save history.')
    parser.add_argument('--target_folder', nargs='?', default='./aggregated_results',
                        help='Option for Conv weight matrix.')
    parser.add_argument('--target_rank', type=int, default=1,
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
    parser.add_argument('--metric', nargs='?', default='val_loss',
                        help='option for metric')
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
    if not os.path.isdir(args.target_folder):
        os.makedirs(args.target_folder)
    df_history.to_csv(os.path.join(args.target_folder,'history_'+str(args.dataset)+'_'+str(args.option)+'.csv'),index=False)


def evaluate_stabilization_rate(args):
    print('evaluate stabilization')
    global options
    global metrics
    observation_size = 40
    mean_rec = list()
    median_rec = list()
    std_rec = list()
    for option in options:
        if option == 'h_rank':
            target_rank = 1
        else:
            target_rank = args.target_rank
        if option in identity_options:
            history_path = './history_updated_identity/'
        else:
            history_path = args.history_path
        target_folder = os.path.join(history_path, args.dataset)
        file = "_".join([option, "kernel_size", str(args.kernel_size),
                          "num_kernel", str(args.kernel),
                          "channel", str(args.channel),
                          "rank",str(target_rank),
                          "epoch", str(args.epoch)])
        target_file = os.path.join(target_folder, file+'.csv')
        df_target = pd.read_csv(target_file)
        df_subset = df_target.iloc[-observation_size:]
        col_mean = df_subset.mean(axis=0)[args.metric]
        col_median = df_subset.median(axis=0)[args.metric]
        col_std = df_subset.std(axis=0)[args.metric]
        mean_rec.append(col_mean)
        median_rec.append(col_median)
        std_rec.append(col_std)
    final_rec = dict()
    final_rec['approach'] = options
    final_rec['mean'] = mean_rec
    final_rec['median'] = median_rec
    final_rec['std'] = std_rec
    df_final_rec = pd.DataFrame(final_rec)
    config = '_'.join(['Last', str(observation_size),'epoches', args.metric,'statistics',
                       'kernel_size',str(args.kernel_size),
                       'rank', str(args.target_rank)])
    results_folder = os.path.join(args.target_folder, args.dataset)
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
    df_final_rec.to_csv(os.path.join(results_folder,config+'.csv'), index=False)
    print(config)




if __name__ == '__main__':
    # Data loading
    args = parse_args()
    # load_history(args)
    evaluate_stabilization_rate(args)

