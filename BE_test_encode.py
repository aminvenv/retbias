from sentence_transformers import SentenceTransformer, util
import os
import csv
import pickle
import time
import faiss
import numpy as np
from tqdm import tqdm 


from tools import load_collection, load_queries, get_query_id_and_text
import argparse
import torch
import json


def main(args):
    model_path = args.model_path
    model = SentenceTransformer(model_path,  device='cuda:'+args.cuda)

    embedding_cache_path = args.embedding_path

    collection = load_collection(args.collection_path)

    if not os.path.exists(embedding_cache_path):
        corpus_sentences = set()
        docids = []
        doc_texts = []
        for key in tqdm(collection.keys(), desc="Saving passage id and passage text ..."):
            docids.append(key)
            doc_texts.append(collection[key])
        corpus_sentences = doc_texts

        print("Encode the corpus. This might take a while")
        corpus_embeddings = model.encode(corpus_sentences,
                                         batch_size=args.batch_size,
                                         show_progress_bar=True,
                                         convert_to_numpy=True)

        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]

        print("Store file on disc")

        with open(embedding_cache_path, 'w') as f:
            for i in tqdm(range(0, len(docids)), desc="writing into jsonl"):
                item = {"id": docids[i],
                        "contents": doc_texts[i],
                         "vector": corpus_embeddings[i].tolist()}
                f.write(json.dumps(item) + "\n")
    else:
        print("Are you sure? the file exists!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--cuda", type=str, required=True)
    parser.add_argument("--embedding_path", type=str)
    parser.add_argument("--collection_path", type=str)
    args = parser.parse_args()

    main(args)
