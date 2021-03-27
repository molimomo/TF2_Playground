import os
import subprocess

datasets = ['mnist']
metrices = ['val_accuracy','val_loss']
target_ranks =[10, 20, 30]
script_name = 'results_aggregation.py'
for dataset in datasets:
    for metric in metrices:
        for target_rank in target_ranks:
            subprocess.call(['python3', script_name,
                            '--dataset=' + dataset,
                            '--metric='+metric,
                            '--agg_option=' + 'multiple_runs',
                            '--target_rank=' + str(target_rank)
                            ])
