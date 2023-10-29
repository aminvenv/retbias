This project works with python==3.7.10.

Use ret_environment.yml to create and setup the conda env.


To install pyserini you can find a complete description [here](https://github.com/castorini/pyserini).


# 1. Data
Download the MS MARCO Passage Ranking collection and queries and extract them into ./msmarco-data path:

```
mkdir msmarco-data

wget https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz -P msmarco-data/
tar xvfz msmarco-data/collection.tar.gz -C msmarco-data/


wget https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz -P msmarco-data/
tar xvfz msmarco-data/queries.tar.gz -C msmarco-data/
```
The `msmarco-data` folder will contain the following files:

1. collection.tsv
2. queries.dev.tsv
3. queries.eval.tsv
4. queries.train.tsv


Download qrels:

```
wget https://msmarco.blob.core.windows.net/msmarcoranking/qrels.train.tsv -P msmarco-data/
```


Download the shuffled msmarco training triplets (query, positive passage, negative passage) :

```
wget 'https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz' -P msmarco-data/
tar xvfz msmarco-data/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz' -C msmarco-data/
```
Download validation triplets:
```
wget https://sbert.net7/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz -P msmarco-data/
tar xvfz msmarco-data/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz -C msmarco-data/
```
Note:
`msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz` and `msmarco-qidpidtriples.rnd-shuf.train.tsv.gz` is a randomly
shuffled version of qidpidtriples.train.full.2.tsv.gz from the MS Marco website [1].
We follow the [Sentence-Transformer](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/ms_marco/train_cross-encoder_scratch.py) which extracted
in the train-eval split 500 random queries that can be used for evaluation during training [1]

We sample 50 positive passages for each query from `msmarco-qidpidtriples.rnd-shuf.train.tsv.gz` using:
```commandline
python get_msmarco_triplets.py
```
The output will be a smaller version of shuffled triplets named `msmarco-qidpidtriples.rnd.shuf.small50.from-all-train.tsv.gz`
### Query Sets
`queries.train.tsv` includes about 800,000 queries, but only 502939 queries have at least one relevant document.
we clean it using msmarco.ipynb to have only 502939. Out of these 502,939, we extract the validation queries `dev500_qids.tsv` (as already mentioned above, we use the same queries as provided in [1], i.e., queries which are in `msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz`, and as already mentioned above)
and exclude those queries from 503,939. Then we divide the queries into two half part each of which includes 250k queries.

To create OTQ we sample 250k queries out of more than 500k queries in the training set:
```commandline
python get_trainingQ_and_OTQ.py
```
#### Training data and OTQ
The output will be in `250k-train-queries.train.tsv` (for training) and `250k-eval-queries.train.tsv`(OTQ) in `msmarco-data` folder.

We then prepare training data files with different training query set sizes 50k, 100k, ..., 250k ([50000] and [50k] should be modified accordingly)
```
python prepare_xk.py --num_of_queries 50000 --name 50k 
```
The output will be 5 files in `msmarco-data` folder with names of `msmarco-qidpidtriples.rnd-shuf.small50-en-251ktrain-[name].tsv.gz'`, where name can be 50k, 100k, ..., 250k.

####  GeQ
Run the following to prepare the GeQ:
```
python dt5q.py
```
The output will be saved in `./msmarco-data/dt5q/generated_queries.tsv`
# 2. Training
### Bi-Encoder
Use the following to train a Bi-Encoder with [TL] queries ([TL] can be selected out of {50k , 100k, ..., 250k}):
```commandline
python train_be.py --batch_size 64 --epochs 20 --train_letter [TL] --cuda 0
```
trained models will be saved as `models/BE-bert-base-uncased-[TL]-max[max_seq_length]`
### Cross-Encoder
Use the following to train a Cross-Encoder with [TL] queries ([TL] can be selected out of {50k , 100k, ..., 250k}):

```commandline
python train_ce.py --train_letter [TL] --batch_size 32 --cuda 0
```

trained models will be saved as `./models/CE-bert-base-uncased-[TL]`

# 3. Retrieval
## BM25
We use Pyserini toolkit for indexing and retrieval with BM25:

You can find a complete description of the procedure for BM25 [here](https://github.com/castorini/pyserini/blob/master/docs/experiments-msmarco-passage.md).

## Bi-Encoder
======= Encoding

To do the retrieval with bi-encoders we need to first encode the collection using the encoders trained in the above.
We run `BE_test_encode.py` to encode the collection into embedding_path.
```
python BE_test_encode.py \
 --model_path ./models/BE-bert-base-uncased-[TL]-max300/ \
 --batch_size 256 \
 --collection_path ./msmarco-data/collection.tsv  \
 --embedding_path ./msmarco-data/embedding_path/[TL]_bert-base-uncased_300v.embd -\
 -cuda 1
```

The output will be a jsonl file in which each line contains something with the following format:
```{"id": "CACM-2636",
        "contents": "Generation of Random Correlated Normal ... \n",
          "vector": [0.123, -0.004]}
```
======= Indexing

Next, we use pyserini to build the dense indexes with the embeddings in `./msmarco-data/DR_faiss_index/` path.
we run the following code to index the encoded documents using the following command:

```
python -m pyserini.index.faiss
  --input ./msmarco-data/embedding_path/[name/of/the/embedding_file] \
  --output ./msmarco-data/DR_faiss_index/[name/of/the/index] \
  --hnsw
 ```

========Retreival

We can use the following to run the retrieval on the indexes:
```
python -m pyserini.search.faiss   --index ../../msmarco-data/DR_faiss_index/[name/of/the/index] \
    --topics ../../msmarco-data/dt5q/GQ_queries.tsv \
    --encoder ../../models/BE-bert-base-uncased-[TL]-max300/ \
    --output [path/for/the/run/file]  \
    --output-format trec   \
    --batch-size 36 \
     --threads 12 \
    --device cuda:0
```

## Cross-Encoder
To perform retrieval with cross-encoder we re-rank the top-k ranked documents from BM25 as the initial ranker.
Use the following to run inference with a cross-encoder:
```
python CE_test_ce.py --run_path [path/to/bm25/run/file] \
--output_path [path/to/the/output/run/file] \
--query_path ./msmarco-data/dt5q/GQ_queries.tsv \
--collection_path ./msmarco-data/collection.tsv \
--batch_size 64 \
--model_path ./models/CE-bert-base-uncased-[TL]-latest/   \
--cuda 1
```

we need to specify the rank in the run files based on the scores that we get with the code above. To do so we run the following for each run file:

```
python msmarco_convert.py --run_file [path/to/run/file/from/ce/models] 
```
# 4. Retrievability Bias

We use the code from prior work [2] to compute the retrievability bias.
Download and extract the repository into `retb` folder.
Then, we compute the retrievability score for each document:
```
python document_retrievability_calculator.py \
--run_file [path/to/the/run/file] \
--out_file [path/to/ret/score/output/file] \
--b $b --c $c  \
--collection ../msmarco-data/collection.tsv 
```

Next we compute the gini using the following:
```
python gini_calculator.py --ret_file [path/to/a/ret/score/output/file]
```


[1] [Sentence-Transformer Library](https://www.sbert.net/)

[2] [Retrievability Bias Repository](https://github.com/leifos/retrievability)
