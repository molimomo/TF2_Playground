import pandas as pd
import numpy as np
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
approaches_a = ['l_rank', 'HL', 'l_rank_identity', 'HL_identity']
approaches_b = ['dual_low_z_v', 'dual_low_zi_v', 'dual_low_z_vi', 'dual_low_zi_vi']
metrices = ['val_loss', 'val_accuracy', 'loss', 'accuracy']

def parse_args():
    parser = argparse.ArgumentParser(description="Plotting results.")
    parser.add_argument('--option', nargs='?', default='varying_rank',
                        help='option for plotting results')
    parser.add_argument('--metric', nargs='?', default='val_loss',
                        help='option for metric')
    parser.add_argument('--group', nargs='?', default='a',
                        help='option for group')
    parser.add_argument('--target_ranks', nargs='?', default='[1, 2, 3]',
                        help='Target ranks for plotting.')
    parser.add_argument('--dataset', nargs='?', default='mnist',
                        help='Choose a Dataset.')
    parser.add_argument('--history_path', nargs='?', default='./history/',
                        help='The folder path to save history.')
    parser.add_argument('--figure_path', nargs='?', default='./result_figures/',
                        help='The folder path to save result figures.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Training Epoch.')
    parser.add_argument('--channel', type=int, default=1,
                        help='Number of channel.')
    parser.add_argument('--kernel', type=int, default=1,
                        help='Number of kernel.')
    parser.add_argument('--kernel_size',  type=int, default=3,
                        help='The dimension for a kernel.')

    return parser.parse_args()

def plot_varying_target_ranks(args):
    global approaches
    target_folder = os.path.join(args.history_path, args.dataset)
    if args.kernel_size == 3:
        target_ranks = [1, 2, 3]
    elif args.kernel_size == 5:
        target_ranks = [1, 3, 5]
    elif args.kernel_size == 7:
        target_ranks = [1, 3, 5, 7]

    # Fetch high rank approach
    file = '_'.join(['h_rank',
                     'kernel_size', str(args.kernel_size),
                     'num_kernel', str(args.kernel),
                     'channel', str(args.channel),
                     'rank', '1',
                     'epoch', str(args.epoch)])
    h_rank_file = os.path.join(target_folder,file+'.csv')
    pd_h_rank = pd.read_csv(h_rank_file)
    h_rank_rec = pd_h_rank[args.metric].values

    # Fetch target rank related approaches
    if args.group == 'a':
        approaches = approaches_a
    elif args.group == 'b':
        approaches = approaches_b
    if args.metric == 'val_loss':
        ylim = [0, 1]
    else:
        ylim = [0.7, 1]

    fig, axes = plt.subplots(nrows=2, ncols=2)
    for appro_idx in range(len(approaches)):
        per_approach = list()
        ranks_str = list()
        for rank in target_ranks:
            ranks_str.append('rank='+str(rank))
            file = '_'.join([approaches[appro_idx],
                            'kernel_size', str(args.kernel_size),
                            'num_kernel',str(args.kernel),
                            'channel', str(args.channel),
                            'rank',str(rank),
                            'epoch',str(args.epoch)])
            print(file)
            target_file = os.path.join(target_folder,file +'.csv')
            print(target_file)
            pd_target = pd.read_csv(target_file)
            per_approach.append(pd_target[args.metric].values)

        metric = np.array(per_approach).reshape(len(target_ranks), int(args.epoch)).T
        df_approach_metric = pd.DataFrame(metric, columns=ranks_str, index=range(args.epoch))
        df_approach_metric['h_rank'] = h_rank_rec
        style = list()
        for i in range(len(df_approach_metric.columns)):
            if i == len(df_approach_metric.columns) - 1:
                style.append('k:')
            else:
                style.append('-')
        lines = df_approach_metric.plot.line(ylim=ylim, ax=axes[int(appro_idx/2), int(appro_idx%2)], style=style)
        lines.set(xlabel='Epoch', ylabel=args.metric,title=approaches[appro_idx]+' - '+args.metric)

    res_config = 'vary_target_rank_'+ str(args.metric) + '_kernel_size_'+str(args.kernel_size)+'_group_'+str(args.group)+'.png'
    figure_folder = os.path.join(args.figure_path, args.dataset)
    if not os.path.isdir(figure_folder):
        os.mkdir(figure_folder)
    plt.tight_layout()
    plt.savefig(os.path.join(figure_folder, res_config))
    plt.show()

    print('dsds')


def polt_line_figure():
    print('sdsd')

if __name__ == '__main__':
    # Data loading
    args = parse_args()
    plot_varying_target_ranks(args)


