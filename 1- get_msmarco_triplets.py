import pandas as pd
import numpy as np
from tqdm import tqdm

from tools import load_queries


def main():
    fullv2 = pd.read_csv('msmarco-data/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz', 
                        compression='gzip', names=['qid', 'ppid', 'npid'], sep='\t',
                        quotechar='"')

    small_triplets = fullv2.groupby(['qid','ppid']).apply(lambda x: x.sample(n=50, random_state=1) if len(x)>50 else x).reset_index(drop=True)
    small_triplets.to_csv("./msmarco-data/msmarco-qidpidtriples.rnd.shuf.small50.from-all-train.tsv.gz", 
               index=False, 
               header=False,
               sep='\t',
               compression="gzip")
    

if __name__ == '__main__':
    main()
