import os
import argparse
import json
import pandas as pd
import numpy as np
import sys

options = ["h_rank",
           "l_rank",
           "l_rank_identity",
           "HL",
           "HL_identity"]

identity_options = ["l_rank_identity",
                    "HL_identity",
                    "dual_low_zi_v",
                    "dual_low_z_vi",
                    "dual_low_zi_vi"
                    ]

metrics = ["loss", "accuracy", "val_loss","val_accuracy"]

def parse_args():
    parser = argparse.ArgumentParser(description="Run aggregation.")
    parser.add_argument('--agg_option', nargs='?', default='multiple_runs',
                        help='Option for aggregation.')
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
    parser.add_argument('--observation_size', type=int, default=40,
                        help='observe last number of epoches.')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='The dimension for a kernel.')
    parser.add_argument('--metric', nargs='?', default='val_loss',
                        help='option for metric')
    return parser.parse_args()

def evaluate_stabilization_rate(args):
    print('evaluate stabilization')
    global options
    global metrics
    observation_size = args.observation_size
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

def hit_checking(mean, std, range, val, is_loss):
    if range == 0:
        if is_loss:
            return val <= mean
        else:
            return val >= mean
    else:
        boundary_min = max([mean - (range * std), 0])
        if is_loss:
            boundary_max = mean + (range * std)
        else:
            boundary_max = min([mean + (range * std), 1])
        return (val <= boundary_max) and (val >= boundary_min)

def evaluate_converge_rate(args):
    # Fetch last 40 epoch statistic
    stablilization_config = '_'.join(['Last', str(args.observation_size),'epoches', args.metric,'statistics',
                       'kernel_size',str(args.kernel_size),
                       'rank', str(args.target_rank)])
    stablilization_folder = os.path.join(args.target_folder, args.dataset)
    stabilization_file = os.path.join(stablilization_folder, stablilization_config+'.csv')
    df_stable = pd.read_csv(stabilization_file, index_col='approach')
    hit_list = list()
    for option in options:
        # Fetch all 100 epoch history
        if option in identity_options:
            all_history_folder = os.path.join('history_updated_identity', args.dataset)
        else:
            all_history_folder = os.path.join(args.history_path, args.dataset)
        if option == 'h_rank':
            history_config = "_".join([option, "kernel_size", str(args.kernel_size),
                          "num_kernel", str(args.kernel),
                          "channel", str(args.channel),
                          "rank",str(1),
                          "epoch", str(args.epoch)]) + '.csv'
        else:
            history_config = "_".join([option, "kernel_size", str(args.kernel_size),
                                       "num_kernel", str(args.kernel),
                                       "channel", str(args.channel),
                                       "rank", str(args.target_rank),
                                       "epoch", str(args.epoch)]) + '.csv'

        print(history_config)
        history_file = os.path.join(all_history_folder , history_config)
        df_history = pd.read_csv(history_file, usecols=[str(args.metric)])
        hit_flags = [False, False, False, False]
        hit_rec = np.zeros(len(hit_flags))
        mean = df_stable['mean'][option]
        std = df_stable['std'][option]
        is_loss = (args.metric == 'val_loss')
        for idx, row in df_history.iterrows():
            if all(hit_flags):
                break
            for i in range(len(hit_flags)):
                if not hit_flags[i]:
                    hit_flags[i] = hit_checking(mean, std, i, row[args.metric], is_loss)
                    if hit_flags[i]:
                        hit_rec[i] = idx
        hit_list.append(hit_rec)
    df_hit_rec = pd.DataFrame(hit_list, index=options)
    df_hit_rec.columns = ["0_sigma", "1_sigma", "2_sigma", "3_sigma"]
    df_final = df_stable.merge(df_hit_rec, left_index=True, right_index=True)
    config = '_'.join(['Converge_rate_Last', str(args.observation_size), 'epoches', args.metric, 'statistics',
                       'kernel_size', str(args.kernel_size),
                       'rank', str(args.target_rank)])
    df_final.to_csv(os.path.join(args.target_folder, args.dataset, config+'.csv'))

def evaluate_afterward(args):
    print('check afterward epoch')
    global options
    # read file
    loss_rec = list()
    accuracy_rec = list()
    loss_idx_rec = list()
    accuracy_idx_rec = list()
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
        max_accuracy = -1
        min_loss = sys.maxsize
        max_acc_idx = 0
        min_loss_idx = 0
        accuracy_cnt = 0
        loss_cnt = 0
        accuracy_flag = False
        loss_flag = False

        for idx, row in df_target.iterrows():
            val_loss = row['val_loss']
            val_accuracy = row['val_accuracy']

            # Check accuracy
            if accuracy_flag == False:
                if max_accuracy < val_accuracy:
                    max_accuracy = val_accuracy
                    accuracy_cnt = 0
                    max_acc_idx = idx
                else:
                    accuracy_cnt = accuracy_cnt + 1
                if accuracy_cnt == 5:
                    accuracy_flag = True

            # Check loss
            if loss_flag == False:
                if min_loss > val_loss:
                    min_loss = val_loss
                    loss_cnt = 0
                    min_loss_idx = idx
                else:
                    loss_cnt = loss_cnt + 1
                if loss_cnt == 5:
                    loss_flag = True


            if loss_flag and accuracy_flag:
                break

        accuracy_rec.append(max_accuracy)
        accuracy_idx_rec.append(max_acc_idx)
        loss_rec.append(min_loss)
        loss_idx_rec.append(min_loss_idx)
    tmp_rec = dict()
    tmp_rec['approach'] = options
    tmp_rec['max_accuracy'] = accuracy_rec
    tmp_rec['max_accuracy_epoch'] = accuracy_idx_rec
    tmp_rec['min_loss'] = loss_rec
    tmp_rec['min_loss_epoch'] = loss_idx_rec
    df_rec = pd.DataFrame.from_dict(tmp_rec)
    config = "_".join(["converge_afterward","kernel_size", str(args.kernel_size),
                       "num_kernel", str(args.kernel),
                       "channel", str(args.channel),
                       "rank", str(args.target_rank),
                       "epoch", str(args.epoch)])
    results_folder = os.path.join(args.target_folder, args.dataset)
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
    df_rec.to_csv(os.path.join(results_folder, config + '.csv'), index=False)
    print('done... ' + str(config))

def evaluate_multiple_runs(args):
    history_folder = os.path.join(args.history_path, args.dataset)
    aggregation_folder = os.path.join(args.target_folder, args.dataset)
    if not os.path.exists(aggregation_folder):
        os.makedirs(aggregation_folder)
    for root, directories, files in os.walk(history_folder):
        if len(directories) != 0:
            avg_rec = dict()
            std_rec = dict()
            for option in options:
                history_rec = dict()
                if option == 'h_rank':
                    rank = 1
                else:
                    rank = args.target_rank
                config = "_".join(["kernel_size", str(args.kernel_size),
                                   "num_kernel", str(args.kernel),
                                   "channel", str(args.channel),
                                   "rank", str(rank),
                                   "epoch", str(args.epoch)])
                for d in directories:
                    seed_folder = os.path.join(history_folder, d)
                    target_file = os.path.join(seed_folder, option+'_'+config+'.csv')
                    df_history = pd.read_csv(target_file)
                    history_rec[d] = df_history[args.metric].values
                df_res = pd.DataFrame.from_dict(history_rec)
                option_folder = os.path.join(aggregation_folder, option)
                if not os.path.exists(option_folder):
                    os.makedirs(option_folder)
                df_res.to_csv(os.path.join(option_folder, str(args.metric)+'_multi_runs_'+ str(option) + '_'+config+'.csv'))
                avg_res = df_res.mean(axis=1)
                std_res = df_res.std(axis=1)
                avg_rec[option] = avg_res
                std_rec[option] = std_res
            df_avg_rec = pd.DataFrame.from_dict(avg_rec)
            df_std_rec = pd.DataFrame.from_dict(std_rec)

            df_avg_rec.to_csv(os.path.join(aggregation_folder, 'avg_' + str(args.metric) + '_multi_runs_' + config + '.csv'))
            df_std_rec.to_csv(os.path.join(aggregation_folder, 'std_' + str(args.metric) + '_multi_runs_' + config + '.csv'))
            print('dsd')

def run_aggregation(args):
    agg_option = args.agg_option
    if agg_option == 'converge_rate':
        evaluate_converge_rate(args)
    elif agg_option == 'stablization_rate':
        evaluate_stabilization_rate(args)
    elif agg_option == 'stablization_afterward':
        evaluate_afterward(args)
    elif agg_option == 'multiple_runs':
        evaluate_multiple_runs(args)


if __name__ == '__main__':
    args = parse_args()
    run_aggregation(args)

