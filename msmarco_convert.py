import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm


def main(args):
    file_name = args.run_file
    run_df = pd.read_csv(file_name, 
                       names=['qid', 'Q0', 'docid', 'rank', 'score', 'name'], sep=' ')

    run_df['rank'] = run_df.groupby(['qid'])['score'].rank(method='first', ascending=False).astype(int)
    run_df.sort_values(['qid','rank'], ascending=[True, True], inplace=True)
    run_df.to_csv(file_name+'.sorted', sep=' ', index=False, header=False)
    print('successfully finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
