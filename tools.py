from tqdm import tqdm
import re
import os
import gzip
import os
import tarfile
import logging
import torch
import pandas as pd
from sentence_transformers import InputExample
from sentence_transformers import LoggingHandler, util


def get_query_id_and_text(query_path):
    qid_list = []
    text_list = []
    with open(query_path, 'r') as f:
        for line in tqdm(f, desc="loading query...."):
            qid, text = line.strip().split("\t")
            qid_list.append(qid)
            text_list.append(text)
    return qid_list, text_list


def load_qrel(qrel_path):
    qrel = {}
    with open(qrel_path, 'r') as f:
        for line in tqdm(f, desc="loading qrel...."):
            qid, _, docid, label = line.strip().split(" ")
            if qid not in qrel.keys():
                qrel[qid] = {}
            qrel[qid][docid] = int(label)
    return qrel


def load_biencoder_training_samples(train_filepath, queries):
    train_queries = {}
    with gzip.open(train_filepath, 'rt') as fIn:
        for line in tqdm(fIn):
            qid, pos_id, neg_id = line.strip().split()

            #Get the positive passage ids
            pos_pids = pos_id
            if qid not in train_queries:
                train_queries[qid] = {'qid':qid, 'query':queries[qid], 'pos':set(), 'neg':set()}
            train_queries[qid]['pos'].add(pos_id)
            train_queries[qid]['neg'].add(neg_id)
    return train_queries


def load_msmarco_training_samples(train_filepath, queries, corpus, dev_samples, pos_neg_ration):
    train_samples = []
    # if not os.path.exists(train_filepath):
    #     logging.info("Download "+os.path.basename(train_filepath))
    #     util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz', train_filepath)

    cnt = 0
    with gzip.open(train_filepath, 'rt') as fIn:
        for line in tqdm(fIn, unit_scale=True):
            qid, pos_id, neg_id = line.strip().split()

            if qid in dev_samples:
                continue

            query = queries[qid]
            if (cnt % (pos_neg_ration+1)) == 0:
                passage = corpus[pos_id]
                label = 1
            else:
                passage = corpus[neg_id]
                label = 0

            train_samples.append(InputExample(texts=[query, passage], label=label))
            cnt += 1

            # if cnt >= max_train_samples:
            #     break
    return train_samples


def load_msmarco_dev_samples(train_eval_filepath, num_dev_queries, num_max_dev_negatives, corpus, queries):
    dev_samples = {}
    with gzip.open(train_eval_filepath, 'rt') as fIn:
        for line in fIn:
            qid, pos_id, neg_id = line.strip().split()
            
            if qid not in dev_samples and len(dev_samples) < num_dev_queries:
                dev_samples[qid] = {'query': queries[qid], 'positive': set(), 'negative': set()}

            if qid in dev_samples:
                dev_samples[qid]['positive'].add(corpus[pos_id])

                if len(dev_samples[qid]['negative']) < num_max_dev_negatives:
                    dev_samples[qid]['negative'].add(corpus[neg_id])
    return dev_samples


def load_run(run_path, run_type='trec'):
    run = {}
    with open(run_path, 'r') as f:
        for line in tqdm(f, desc="loading run...."):
            if run_type == 'msmarco':
                qid, docid, score = line.strip().split("\t")
            elif run_type == 'trec':
                qid, _, docid, rank, score, _ = line.strip().split(" ")
            qid = qid
            docid = docid
            if qid not in run.keys():
                run[qid] = []
            run[qid].append(docid)
    return run


def wapo_load_collection(collection_path):
    collection = {}
    with open(collection_path, 'r') as f:
        # num = 0
        for line in tqdm(f, desc="loading collection...."):
            try:
                docid, text = line.strip().split("\t")
                collection[docid] = text
            except:
                collection[docid] = "None"
            # num += 1
            # if num == 1025:
            #     break
    return collection



def backup_load_collection(collection_path, tokenizer):
    collection = {}
    with open(collection_path, 'r') as f:
        # num = 0
        for line in tqdm(f, desc="loading collection...."):
            docid, text = line.strip().split("\t")
            tokenized_text = tokenizer.batch_decode(tokenizer(text,
                                                       padding=True,
                                                       truncation=True,
                                                       return_tensors="pt",
                                                       max_length=200)['input_ids'],
                                             skip_special_tokens=True)[0]
            collection[docid] = tokenized_text
            # num+=1
            # if num%500000==0:
            #     print('{} passages already parsed'.format(num))
    return collection

def load_collection(collection_path):
    """
    This one is the original and proper one for reading msmarco collection
    """
    collection = {}
    with open(collection_path, 'r') as f:
        for line in tqdm(f, desc="loading collection...."):
            docid, text = line.strip().split("\t")
            collection[docid] = text
    return collection



def download_msmarco_queries(data_folder):
    queries_filepath = os.path.join(data_folder, 'queries.train.tsv')
    if not os.path.exists(queries_filepath):
        tar_filepath = os.path.join(data_folder, 'queries.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download queries.tar.gz")
            util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz', tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)
    return True

def download_msmarco_collection(data_folder):
    collection_filepath = os.path.join(data_folder, 'collection.tsv')
    if not os.path.exists(collection_filepath):
        tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
        if not os.path.exists(tar_filepath):
            logging.info("Download collection.tar.gz")
            util.http_get('https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

        with tarfile.open(tar_filepath, "r:gz") as tar:
            tar.extractall(path=data_folder)    
    
def load_queries(query_path):
    query = {}
    with open(query_path, 'r') as f:
        for line in tqdm(f, desc="loading query...."):
            qid, text = line.strip().split("\t")
            query[qid] = text
    return query


def load_dev_queries(dev_query_path):
    # query_path = './msmarco-data/dev500-queries.tsv'
    query = {}
    all_queries = pd.read_csv('./msmarco-data/clean502939-queries.train.tsv', sep='\t',
                              names=['qid', 'qtext'])
    with open(dev_query_path, 'r') as f:
        for line in tqdm(f, desc="loading query...."):
            qid = line.strip()
            text = all_queries[all_queries['qid']==int(qid)].qtext.values[0]
            query[qid] = text

    return query

def get_batch_text(start, end, docids, collection):
    batch_text = []
    for docid in docids[start: end]:
        batch_text.append(collection[docid])
    return batch_text

def get_run_text(run, queries, collection):
    run_text = []
    run_id = []
    for qid in tqdm(run):
        for docid in run[qid]:
            if docid in collection: # this should be removed 
                run_text.append([queries[qid], collection[docid]])
                run_id.append([qid, docid])
    return run_text, run_id
        
def get_ExactMatch_run_text(run, queries, collection):
    run_text = []
    run_id = []
    for qid in tqdm(run):
        EM_query = queries[qid]
        query_tokens = EM_query.lower().split(" ")
        for docid in run[qid]:
            if docid in collection: # this should be removed 
                sent = collection[docid]
                EM_doc = ""
                sent_tokens = sent.split(" ")
                for token in sent_tokens:
                    if token.lower() in query_tokens:
                        EM_doc = EM_doc + " " + token
                    else:
                        EM_doc = EM_doc + " [MASK]"
                run_text.append([EM_query, EM_doc])
                run_id.append([qid, docid])
    return run_text, run_id
            

    
    
def backup_get_run_text(run, queries, collection):
    run_text = []
    run_id = []
    for qid in run:
        for docid in run[qid]:
            run_text.append([queries[qid], collection[docid]])
            run_id.append([qid, docid])
    return run_text, run_id
        
def get_stop_ids(tok):
    # hard code for now, from nltk.corpus import stopwords, stopwords.words('english')
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                      "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                      'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                      'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                      'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                      'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                      'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                      'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                      'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                      'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                      "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                      "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                      'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                      "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
                      'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
    # keep some common words in ms marco questions
    stop_words.difference_update(["where", "how", "what", "when", "which", "why", "who"])

    vocab = tok.get_vocab()
    tokens = vocab.keys()

    stop_ids = []

    for stop_word in stop_words:
        ids = tok(stop_word, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            stop_ids.append(ids[0])

    for token in tokens:
        token_id = vocab[token]
        if token_id in stop_ids:
            continue
        if token == '##s':  # remove 's' suffix
            stop_ids.append(token_id)
        if token[0] == '#' and len(token) > 1:  # skip most of subtokens
            continue
        if not re.match("^[A-Za-z0-9_-]*$", token):  # remove numbers, symbols, etc..
            stop_ids.append(token_id)

    return set(stop_ids)
