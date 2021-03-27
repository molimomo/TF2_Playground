import os
import subprocess

datasets = ['mnist']
metrices = ['val_loss','val_accuracy']
option = 'multi_runs'
measurements =['avg','std']
for dataset in datasets:
    for metric in metrices:
        for measurement in measurements:
            subprocess.call(['python3', 'plotting_results.py',
                             '--option='+option,
                             '--dataset=' + dataset,
                             '--metric='+metric,
                             '--measurement='+str(measurement)])


# datasets = ['mnist', 'fashion_mnist']
# kernel_sizes =[3,5,7]
# metrices = ['val_accuracy']
# arg_list = list()
# arg_list.append('python3')
# script_name = 'results_aggregation.py'
# for dataset in datasets:
#     for kernel_size in kernel_sizes:
#         if kernel_size == 3:
#             target_ranks = [1,2,3]
#         elif kernel_size == 5:
#             target_ranks = [1,3,5]
#         elif kernel_size == 7:
#             target_ranks = [1,3,5,7]
#         for metric in metrices:
#             for target_rank in target_ranks:
#                 subprocess.call(['python3', script_name,
#                                 '--dataset=' + dataset,
#                                 '--kernel_size='+str(kernel_size),
#                                 '--metric='+metric,
#                                 '--agg_option=' + 'stablization_afterward',
#                                 '--target_rank=' + str(target_rank)
#                                 ])