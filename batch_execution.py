import os
import subprocess

datasets = ['mnist', 'fashion_mnist']
metrices = ['val_accuracy','val_loss']
kernel_sizes =[3,5,7]
script_name = 'results_aggregation.py'
for dataset in datasets:
    for kernel_size in kernel_sizes:
        if kernel_size == 3:
            target_ranks = [1,2,3]
        elif kernel_size == 5:
            target_ranks = [1,3,5]
        elif kernel_size == 7:
            target_ranks = [1,3,5,7]
        for metric in metrices:
            for target_rank in target_ranks:
                subprocess.call(['python3', script_name,
                                '--dataset=' + dataset,
                                '--kernel_size='+str(kernel_size),
                                '--metric='+metric,
                                '--agg_option=' + 'multiple_runs',
                                '--target_rank=' + str(target_rank)
                                ])
