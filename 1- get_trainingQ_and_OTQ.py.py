import pandas as pd
import numpy as np
from tqdm import tqdm
import gzip
from tools import load_queries


def take_devsamples_qids():
    uniq_qids = []
    with gzip.open('./msmarco-data/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz', 'rt') as fIn:
        for line in fIn:
            qid, pos_id, neg_id = line.strip().split()
            uniq_qids.append(qid)
    return set(uniq_qids)


def main():
    train_qrel_df = pd.read_csv('./msmarco-data/qrels.train.tsv', sep='\t',
                                names=['qid', 'zero', 'pid', 'label'],
                                dtype={'qid':str})

    second_clean_from_triplets = pd.read_csv('./msmarco-data/msmarco-qidpidtriples.rnd.shuf.small50.from-all-train.tsv.gz',
                        compression='gzip', names=['qid', 'ppid', 'npid'], sep='\t',
                        dtype={'qid':str},
                        quotechar='"')
    second_clean_qids = (set(second_clean_from_triplets['qid'].values))
    queries = load_queries('./msmarco-data/queries.train.tsv')
    clean_qids = set(train_qrel_df['qid'].values)
    dev500_qids = take_devsamples_qids()
    pd.DataFrame(dev500_qids).to_csv('./msmarco-data/dev500-queries.tsv',
                                     sep='\t',
                                     header=False,\
                                     index=False)

    clean_qid_qtext = pd.DataFrame(columns=['qid', 'query_text'])
    second_clean_qid_qtext = pd.DataFrame(columns=['qid', 'query_text'])

    qids = []
    qtexts = []

    second_qids = []
    second_qtexts = []

    for query_id in tqdm(queries):
        if query_id in clean_qids:
            qids.append(query_id)
            qtexts.append(queries[query_id])

        if query_id in second_clean_qids:
            second_qids.append(query_id)
            second_qtexts.append(queries[query_id])
    clean_qid_qtext['qid'] = qids
    clean_qid_qtext['query_text'] = qtexts

    second_clean_qid_qtext['qid'] = second_qids
    second_clean_qid_qtext['query_text'] = second_qtexts

    clean_qid_qtext_without_dev500 = clean_qid_qtext[~clean_qid_qtext['qid'].isin(dev500_qids)]
    second_clean_qid_qtext_without_dev500 = second_clean_qid_qtext[~second_clean_qid_qtext['qid'].isin(dev500_qids)]
    train_part_50 = second_clean_qid_qtext_without_dev500.sample(n = 250000)
    eval_rest_part_50 = clean_qid_qtext_without_dev500[~clean_qid_qtext_without_dev500['qid'].isin(set(train_part_50['qid'].values))].sample(n = 250000)

    clean_qid_qtext.to_csv('./msmarco-data/clean502939-queries.train.tsv', index=False, sep='\t', header=False)
    second_clean_qid_qtext.to_csv('./msmarco-data/second_clean502939-queries.train.tsv', index=False, sep='\t', header=False)

    train_part_50.to_csv('./msmarco-data/250k-train-queries.train.tsv', index=False, sep='\t', header=False)
    eval_rest_part_50.to_csv('./msmarco-data/250k-eval-queries.train.tsv', index=False, sep='\t', header=False)


if __name__ == '__main__':
    main()