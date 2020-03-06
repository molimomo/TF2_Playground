import os
import argparse
import json
import pandas as pd
def parse_args():
    parser = argparse.ArgumentParser(description="Run Deep CNN.")
    parser.add_argument('--option', nargs='?', default='h_rank',
                        help='Option for Conv weight matrix.')
    parser.add_argument('--results_folder', nargs='?', default='./history',
                        help='Option for Conv weight matrix.')
    parser.add_argument('--target_folder', nargs='?', default='./aggregated_results',
                        help='Option for Conv weight matrix.')
    parser.add_argument('--target_rank', type=int, default=2,
                        help='Target rank for Conv low rank weight matrix.')
    parser.add_argument('--dataset', nargs='?', default='mnist',
                        help='Choose a Dataset')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Training Epoch')
    return parser.parse_args()

def load_history(args):
    target_file = os.path.join(args.results_folder,"history_"+str(args.dataset)+"_"+str(args.option)+".json")
    print(target_file)
    history = json.load(target_file)
    df_history = pd.read_json(target_file)
    if not os.path.isdir(args.target_folder):
        os.makedirs(args.target_folder)
    df_history.to_csv(os.path.join(args.target_folder,'history_'+str(args.dataset)+'_'+str(args.option)+'.csv'),index=False)

if __name__ == '__main__':
    # Data loading
    args = parse_args()
    load_history(args)

