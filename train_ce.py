from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
import argparse
# from transformers import AutoTokenizer

from tools import load_collection, load_queries, download_msmarco_queries, download_msmarco_collection
from tools import load_msmarco_training_samples, load_msmarco_dev_samples, load_dev_queries


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def main(args):
    model_name = 'bert-base-uncased'
    train_batch_size = args.batch_size
    num_epochs = 1
    model_save_path = './models/CE-'+model_name+'-'+args.train_letter
    pos_neg_ration = 4
    model = CrossEncoder(model_name, num_labels=1, max_length=250, device='cuda:'+args.cuda)
    data_folder = 'msmarco-data'
    os.makedirs(data_folder, exist_ok=True)

    if True:
        collection_filepath = os.path.join('./msmarco-data/model_collection/'+model_name+'/collection.tsv')

    corpus = {}
    corpus = load_collection(collection_filepath)

    try:
        queries_filepath = os.path.join(data_folder,'new250k-train-queries.train.tsv') # 'queries.train.tsv')
    except:
        print("Cannot download and extract msmarco queries") 

    queries = {}
    queries = load_queries(queries_filepath)
    dev_queries = load_dev_queries('./msmarco-data/dev500-queries.tsv')

    num_dev_queries = 500
    num_max_dev_negatives = 200

    train_eval_filepath = os.path.join('msmarco-data', 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz')
    dev_samples = load_msmarco_dev_samples(train_eval_filepath,
                                           num_dev_queries,
                                           num_max_dev_negatives,
                                           corpus,
                                           dev_queries)

    # Read our training file
    train_filepath = os.path.join(data_folder,
                                  'msmarco-qidpidtriples.rnd-shuf.small'+args.small+'-en-251ktrain-'+args.train_letter+'.tsv.gz' )
    train_samples = load_msmarco_training_samples(train_filepath, queries, corpus, dev_samples, pos_neg_ration)
    # We create a DataLoader to load our train samples
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

    evaluator = CERerankingEvaluator(dev_samples, name='train-eval')
    # Configure the training
    warmup_steps = 5000
    logging.info("Warmup-steps: {}".format(warmup_steps))

    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              epochs=num_epochs,
              evaluation_steps=10000,
              warmup_steps=warmup_steps,
              output_path=model_save_path,
              use_amp=True,
              optimizer_params={'lr': 7e-6},
              weight_decay=0.01)

    model.save(model_save_path+'-latest')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", type=str, required=True)
    parser.add_argument("--train_letter", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--cuda", type=str, required=True)
    args = parser.parse_args()
    main(args)
