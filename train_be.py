import sys
import json
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
from collections import defaultdict
from torch.utils.data import IterableDataset
from tqdm import tqdm
from torch.utils.data import Dataset
import random
import pickle
import argparse


from tools import load_biencoder_training_samples, load_collection, load_queries

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


class MSMARCODataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']

        pos_id = query['pos'].pop(0)    #Pop positive and add at end
        pos_text = self.corpus[pos_id]
        query['pos'].append(pos_id)

        neg_id = query['neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)

        return InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(self.queries)


def main(args):
    model_name = 'bert-base-uncased'

    train_batch_size = args.batch_size
    max_seq_length = args.max_seq_length
    num_epochs = args.epochs

    if True:
        logging.info("Create new SBERT model")
        word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device='cuda:'+args.cuda)

    model_save_path = 'models/BE-'+model_name.replace("/", "-")+'-'+args.train_letter+'-max'+str(max_seq_length)

    data_folder = 'msmarco-data'

    corpus = {}

    logging.info("Read corpus: collection.tsv")
    
    if True:
        collection_filepath = os.path.join(data_folder, 'collection.tsv')
    corpus = load_collection(collection_filepath)

    queries = {}
    queries_filepath = os.path.join(data_folder,'new250k-train-queries.train.tsv') # 'queries.train.tsv')
    queries = load_queries(queries_filepath)

    logging.info("Read negatives train file from bm25 results") # optimus
    train_filepath = os.path.join(data_folder,'msmarco-qidpidtriples.rnd-shuf.small50-en-251ktrain-'+args.train_letter+'.tsv.gz') 
    train_queries = {}
    train_queries = load_biencoder_training_samples(train_filepath, queries)
    logging.info("Train queries: {}".format(len(train_queries)))

    train_dataset = MSMARCODataset(train_queries, corpus=corpus)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=num_epochs,
              warmup_steps=args.warmup_steps,
              use_amp=True,
              checkpoint_path=model_save_path,
              checkpoint_save_steps=len(train_dataloader),
              optimizer_params = {'lr': args.lr},
              )

    # Save the model
    model.save(model_save_path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--cuda", type=str, required=True)
    parser.add_argument("--train_letter", type=str, required=True)
    args = parser.parse_args()

    print(args)
    main(args)
