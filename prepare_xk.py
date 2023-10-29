import pandas as pd
import gzip
import argparse
from tqdm import tqdm


def main(args):
    number_of_queries = args.num_of_queries # 50k, 100k, 150k, 200k, 251k
    train_251k_query_list = pd.read_csv('./msmarco-data/250k-train-queries.train.tsv',
                            sep='\t',
                            names=['qid', 'qtext'], dtype={'qid':str})['qid'].values
    
    set_of_picked_queries = set(train_251k_query_list[0:number_of_queries])
    counter = 0
    q_counter = {}
    with gzip.open('msmarco-data/msmarco-qidpidtriples.rnd-shuf.small50-en-251ktrain-'+args.name+'.tsv.gz', 'wt') as wrigher_f:
        with gzip.open('msmarco-data/sampled_triples/msmarco-qidpidtriples.rnd.shuf.small50.from-all-train.tsv.gz', 'rt') as f:
            for line in tqdm(f):
                #print(line.split())
                qid, p_pid, n_pid = line.split()
                if qid in set_of_picked_queries:
                    counter += 1
                    q_counter[qid] = 1
                    wrigher_f.write(line)

    print('number of new triplets:', counter)
    print('number of total queries', len(q_counter))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_queries", type=int, required=True)
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()
    main(args)
