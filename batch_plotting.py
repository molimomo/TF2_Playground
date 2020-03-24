import os
import subprocess

datasets = ['mnist', 'fashion_mnist']
kernel_sizes =[3,5,7]
groups = ['a','b']
metrices = ['val_loss', 'val_accuracy']

subprocess.call(['python3', 'plotting_results.py', '--dataset=fashion_mnist'])

arg_list = list()
arg_list.append('python3')
arg_list.append('plotting_results.py')
for dataset in datasets:
    for kernel_size in kernel_sizes:
        for metric in metrices:
            for group in groups:
                subprocess.call(['python3', 'plotting_results.py',
                                 '--dataset='+dataset,
                                 '--kernel_size='+str(kernel_size),
                                 '--metric='+metric,
                                 '--group='+group])