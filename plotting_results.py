import pandas as pd
import numpy as np
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
approaches = ['h_rank', 'l_rank', 'HL', 'l_rank_identity', 'HL_identity']
approaches_a = ['l_rank', 'HL', 'l_rank_identity', 'HL_identity']
approaches_b = ['dual_low_z_v', 'dual_low_zi_v', 'dual_low_z_vi', 'dual_low_zi_vi']

metrices = ['val_loss', 'val_accuracy', 'loss', 'accuracy']

def parse_args():
    parser = argparse.ArgumentParser(description="Plotting results.")
    parser.add_argument('--option', nargs='?', default='multi_runs',
                        help='option for plotting results')
    parser.add_argument('--metric', nargs='?', default='val_accuracy',
                        help='option for metric')
    parser.add_argument('--group', nargs='?', default='a',
                        help='option for group')
    parser.add_argument('--target_ranks', nargs='?', default='[1, 2, 3]',
                        help='Target ranks for plotting.')
    parser.add_argument('--dataset', nargs='?', default='mnist',
                        help='Choose a Dataset.')
    parser.add_argument('--history_path', nargs='?', default='./history/',
                        help='The folder path to save history.')
    parser.add_argument('--aggregation_path', nargs='?', default='./aggregated_results/',
                        help='The folder path to save history.')
    parser.add_argument('--history_identity_path', nargs='?', default='./history_updated_identity/',
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
    parser.add_argument('--observation_size', type=int, default=40,
                        help='The number of last epoch for observation')
    parser.add_argument('--identity_option', type=int, default=1,
                        help='0: unscalaring, 1: scalar to 1')
    parser.add_argument('--ylim', nargs='?', default='[0, 1]',
                        help='Boundary for y-axis.')
    parser.add_argument('--measurement', nargs='?', default='std',
                        help='Measurement (avg, std)')

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
        ylim = [0, 5]
    else:
        ylim = [0.5, 0.9]

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

'''
Given a dataset, kernel size and target rank, 
plot results within varying approaches.
'''
def plot_varying_approaches(args):
    print('Plotting varying approaches...')
    global approaches_a, approaches_b

    target_folder = os.path.join(args.history_path, args.dataset)

    if args.kernel_size == 3:
        target_ranks = [1, 2, 3]
    elif args.kernel_size == 5:
        target_ranks = [1, 3, 5]
    elif args.kernel_size == 7:
        target_ranks = [1, 3, 5, 7]

    if args.group == 'a':
        approaches = approaches_a
    elif args.group == 'b':
        approaches = approaches_b
    if args.metric == 'val_loss':
        ylim = [0, 1]
    else:
        ylim = [0.5, 0.9]

    style = list()
    if args.group == 'a':
        style.append('b-')
        style.append('g-')
        style.append('r-')
        style.append('c-')
        style.append('k:')
    else:
        style.append('m-')
        style.append('y-')
        style.append('k-')
        style.append('0.8')
        style.append('k:')

    # fig, axes = plt.subplots(nrows=3, ncols=2)

    # Fetch high rank approach
    file = '_'.join(['h_rank',
                     'kernel_size', str(args.kernel_size),
                     'num_kernel', str(args.kernel),
                     'channel', str(args.channel),
                     'rank', '1',
                     'epoch', str(args.epoch)])
    h_rank_file = os.path.join(target_folder, file + '.csv')
    pd_h_rank = pd.read_csv(h_rank_file)
    h_rank_rec = pd_h_rank[args.metric].values

    for rank in target_ranks:
        tmp_rec = list()
        for appro in approaches:
            file = '_'.join([appro,
                'kernel_size', str(args.kernel_size),
                'num_kernel', str(args.kernel),
                'channel', str(args.channel),
                'rank', str(rank),
                'epoch', str(args.epoch)])
            ### trick for comparing unscalar and scalar
            if args.identity_option == 1:
                target_folder = os.path.join(args.history_identity_path, args.dataset)
                target_file = os.path.join(target_folder, file + '.csv')
                if not os.path.exists(target_file):
                    target_folder = os.path.join(args.history_path, args.dataset)
                    target_file = os.path.join(target_folder, file + '.csv')

                target_file = os.path.join(target_folder, file + '.csv')
                print(target_file)
                pd_target = pd.read_csv(target_file)
                tmp_rec.append(pd_target[args.metric].values)
        df_rec = pd.DataFrame(np.array(tmp_rec).transpose())
        df_rec.columns = approaches
        df_rec['h_rank'] = h_rank_rec

        lines = df_rec.plot.line(ylim=ylim, style=style)
        title = ''
        res_config = ''
        if args.identity_option == 1:
            title = 'Varying approaches, group '+ str(args.group)+', rank='+str(rank)+ '(re-scalaring) - ' + args.metric
            res_config = 'vary_approaches_' + str(args.metric) + '_kernel_size_' + str(
                args.kernel_size) + '_group_' + str(
                args.group) + '_rank_' + str(rank) + '_rescalaring.png'
        else:
            title = 'Varying approaches, group '+ str(args.group)+', rank='+str(rank)+ ' - ' + args.metric
            res_config = 'vary_approaches_' + str(args.metric) + '_kernel_size_' + str(
                args.kernel_size) + '_group_' + str(
                args.group) + '_rank_' + str(rank) + '.png'

        lines.set(xlabel='Epoch', ylabel=args.metric, title=title)
        figure_folder = os.path.join(args.figure_path, args.dataset)
        if not os.path.isdir(figure_folder):
            os.mkdir(figure_folder)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, res_config))
        plt.show()
        print('Done')

def plot_converge_status(args):
    print('Plotting last ' +str(args.observation_size)+' epoch observations...')
    target_folder = os.path.join(args.aggregation_path, args.dataset)
    if args.kernel_size == 3:
        target_ranks = [1, 2, 3]
        #target_ranks = [1]
    elif args.kernel_size == 5:
        target_ranks = [1, 3, 5]
    elif args.kernel_size == 7:
        target_ranks = [1,3, 5, 7]
    bar_rec = list()
    fig, ax = plt.subplots()
    init_x = np.arange(9)
    bar_width = 0.25
    approaches = None
    for i in range(len(target_ranks)):
        config = '_'.join(['Last', str(args.observation_size), 'epoches', args.metric, 'statistics',
                       'kernel_size', str(args.kernel_size),
                       'rank', str(target_ranks[i])])
        print(config)
        target_file = os.path.join(target_folder, config+'.csv')
        df_target = pd.read_csv(target_file)
        approaches = df_target['approach'].values
        # fig = plt.figure()
        plt.bar(x=df_target['approach'],
                      height=df_target['mean'],
                      yerr=df_target['std'],
                      label='rank='+str(target_ranks[i]),
                      width=bar_width)

        plt.legend()
        figure_folder = os.path.join(args.figure_path, args.dataset)
        title = '_'.join(['Last', str(args.observation_size), 'epoches', args.metric, 'statistics',
                       'kernel_size', str(args.kernel_size),'rank',str(target_ranks[i])])
        res_config = title + '.png'
        plt.title(title)
        plt.ylabel(args.metric)
        plt.ylim(bottom=0.8)
        plt.ylim(top=1)
        plt.xticks(rotation=90)
        #plt.xticks([r + bar_width for r in range(9)],approaches)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, res_config))
        plt.show()
    print('dsds')

def plot_converge_afterward(args):
    print('Plotting afterward converge...')
    global approaches_a, approaches_b
    approaches = approaches_a + approaches_b
    target_folder = os.path.join(args.aggregation_path, args.dataset)
    if args.kernel_size == 3:
        target_ranks = [1, 2, 3]
    elif args.kernel_size == 5:
        target_ranks = [1, 3, 5]
    elif args.kernel_size == 7:
        target_ranks = [1, 3, 5, 7]

    markers = [">", "<","*","v","^","+","x","h","s"]
    colors = ["b", "g", "r", "c", "m", "y", "k", "0.75", "0.5"]
    for rank in target_ranks:
        fig, axes = plt.subplots(nrows=2, ncols=1)
        config = "_".join(["converge_afterward", "kernel_size", str(args.kernel_size),
                           "num_kernel", str(args.kernel),
                           "channel", str(args.channel),
                           "rank", str(rank),
                           "epoch", str(args.epoch)])
        target_file = os.path.join(target_folder, config + '.csv')
        df_target = pd.read_csv(target_file)

        # axes[0].scatter(df_target['max_accuracy_epoch'].values,df_target['max_accuracy'].values)
        for idx, row in df_target.iterrows():
            label = row['approach']
            x = row['max_accuracy_epoch']
            y = row['max_accuracy']
            axes[0].scatter(x,y, label=label, c=colors[idx], marker=markers[idx])
            # axes[0].annotate(label, (x,y), textcoords="offset points", xytext=(0,-10), ha='center')
        title = '_'.join(['Accuracy',
                          'kernel_size', str(args.kernel_size), 'rank', str(rank), 'Higher', 'the better'])
        axes[0].set_title(title)
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlabel('Epoch')

        # axes[1].scatter(df_target['min_loss_epoch'].values,df_target['min_loss'].values,c='red')
        for idx, row in df_target.iterrows():
            label = row['approach']
            x = row['min_loss_epoch']
            y = row['min_loss']
            axes[1].scatter(x, y, label=label, c=colors[idx], marker=markers[idx])
        title = '_'.join(['Loss',
                          'kernel_size', str(args.kernel_size), 'rank', str(rank), 'Lower','the better'])
        axes[1].set_title(title)
        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        handles, labels = axes[1].get_legend_handles_labels()
        lg=fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.25,0.5))
        plt.tight_layout()
        fig_title = '_'.join(['Converge_afterward',
                          'kernel_size', str(args.kernel_size), 'rank', str(rank)])
        figure_folder = os.path.join(args.figure_path, args.dataset)
        res_config = fig_title+'.png'
        plt.savefig(os.path.join(figure_folder, res_config),
                    format='png',
                    bbox_extra_artists=(lg,),
                    bbox_inches='tight')
        plt.show()

        print('dsds')

def plot_multi_runs(args):
    target_folder = os.path.join(args.aggregation_path, args.dataset)

    if args.kernel_size == 3:
        target_ranks = [1, 2, 3]
    elif args.kernel_size == 5:
        target_ranks = [1, 3, 5]
    elif args.kernel_size == 7:
        target_ranks = [1, 3, 5, 7]

    if args.metric == 'val_loss':
        ylim = [0, 5]
    else:
        ylim = [0.65, 0.95]

    for rank in target_ranks:
        config = '_'.join([args.metric,'multi_runs',
                              'kernel_size',str(args.kernel_size),
                              "num_kernel", str(args.kernel),
                              "channel", str(args.channel),
                              "rank", str(rank),
                              "epoch", str(args.epoch)])
        target_file = os.path.join(target_folder,args.measurement+'_'+config+'.csv')
        df_target = pd.read_csv(target_file, usecols=approaches)
        if args.measurement == 'avg':
            lines = df_target.plot.line(ylim=ylim)
        else:
            lines = df_target.plot.line()
        title = '_'.join([args.measurement,args.metric,
                          'kernel_size', str(args.kernel_size), 'rank', str(rank)])
        lines.set(xlabel='Epoch', ylabel=args.measurement+'_'+args.metric, title=title)
        figure_folder = os.path.join(args.figure_path, args.dataset)
        if not os.path.exists(figure_folder):
            os.makedirs(figure_folder)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_folder, title+'.png'))
        plt.show()
        print('dsdsds')




if __name__ == '__main__':
    # Data loading
    args = parse_args()
    if args.option == 'varying_rank':
        plot_varying_target_ranks(args)
    elif args.option == 'varying_approach':
        plot_varying_approaches(args)
    elif args.option == 'converge_status':
        plot_converge_status(args)
    elif args.option == 'afterward_converge':
        plot_converge_afterward(args)
    elif args.option == 'multi_runs':
        plot_multi_runs(args)




