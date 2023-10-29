from sentence_transformers import SentenceTransformer, util
from sentence_transformers import CrossEncoder
import os
import csv
import pickle
import time
import sys
from tqdm import tqdm

import torch

import argparse
from tools import get_stop_ids, load_run, load_collection, get_batch_text, load_queries, get_run_text # most of credit goes to IELAB/TILDE


def main(args):
    model_path = args.model_path
    cross_encoder_model = CrossEncoder(model_path, max_length=250, device='cuda:'+args.cuda)
    
    collection = load_collection(args.collection_path)
    queries = load_queries(args.query_path) # should be 251-eval-queries.tsv
    
    if args.run_path:
        run = load_run(args.run_path)
        run_pair_text, run_pair_id = get_run_text(run, queries, collection)
    


    start_time = time.time()
    Gbatch_size = 256000
    last_batch_id = int(len(run_pair_text)/Gbatch_size)-1
    with open(args.output_path, 'w') as f:
        for i in tqdm(range(0, int(len(run_pair_text)/Gbatch_size))):
            if i==last_batch_id:
                current_batch = run_pair_text[i*Gbatch_size:]
                current_batch_id = run_pair_id[i*Gbatch_size:]
            else:
                current_batch = run_pair_text[i*Gbatch_size:(i+1)*Gbatch_size]
                current_batch_id = run_pair_id[i*Gbatch_size:(i+1)*Gbatch_size]

            if cross_encoder_model.config.num_labels > 1:
                ce_scores = cross_encoder_model.predict(current_batch, 
                                                           show_progress_bar=True,
                                                           batch_size=args.batch_size,
                                                           apply_softmax=True)[:, 1].tolist()
            else:
                ce_scores = cross_encoder_model.predict(current_batch, 
                                                           show_progress_bar=True,
                                                           batch_size=args.batch_size).tolist()

            for j in range(len(current_batch)):
                f.write('{} Q0 {} 0 {} cross-encoder\n'.format(current_batch_id[j][0], current_batch_id[j][1],ce_scores[j]))
            print('Gbatch {} finished'.format(i))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--query_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--cuda", type=str, required=True)
    parser.add_argument("--collection_path", type=str)
    args = parser.parse_args()
    save_dir = os.path.dirname(args.output_path)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    main(args)
