import os
import subprocess

datasets = ['fashion_mnist']
kernel_sizes =[3,5,7]
metrices = ['val_loss']

arg_list = list()
arg_list.append('python3')
arg_list.append('LRCNN.py')

for dataset in datasets:
    for kernel_size in kernel_sizes:
        for metric in metrices:
            subprocess.call(['python3', 'plotting_results.py',
                             '--dataset=' + dataset,
                             '--kernel_size='+str(kernel_size),
                             '--metric='+metric])

