from tqdm import tqdm
import random
from tools import load_collection
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
    model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
    model.to(device)
    collection = load_collection('./msmarco-data/collection.tsv')
    picked_docids = random.sample(list(collection.keys()), 250000)
    with open('./msmarco-data/dt5q/picked_docid.tsv', 'w') as f:
        for row in picked_docids:
            f.write(row+'\n')

    with open('./msmarco-data/dt5q/generated_queries.tsv', 'w') as f:
        id_counter = 9000000
        for doc_id in tqdm(picked_docids):
            doc_text = collection[doc_id]
            input_ids = tokenizer.encode(doc_text, return_tensors='pt').to(device)
            outputs = model.generate(
                input_ids=input_ids,
                max_length=32,
                do_sample=True,
                top_k=10,
                num_return_sequences=1,)
            new_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
            f.write(str(id_counter)+'\t'+str(doc_id)+'\t'+new_query+'\n')
            id_counter += 1


if __name__ == '__main__':
    main()
