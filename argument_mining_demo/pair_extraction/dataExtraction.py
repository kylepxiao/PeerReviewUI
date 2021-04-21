import os
import ast
import pandas as pd
import allennlp_models.pair_classification
from allennlp.predictors.predictor import Predictor
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from transformers import pipeline
from collections import defaultdict

def extract_to_file(tokenized_reviews, output_path):
    review_pair_id = 0
    with open(os.path.join(output_path, 'ReviewRebuttalPredict.txt'), 'w') as output_file:
        for i in range(len(tokenized_reviews)):
            # Get dummy labels for all sentences in a review
            review = tokenized_reviews[i]
            for sentence in review:
                if sentence == ' ' or sentence == '\t':
                    continue
                row = ''
                row += sentence.replace('\n', '[line_break_token]') + '\t'
                row += 'O' + '\t'
                row += 'O' + '\t'
                row += 'Review' + '\t'
                row += str(review_pair_id)
                row += '\n'
                output_file.write(row)
            #output_file.write('\n')
            # Get all combinations of reviews with other reviews
            for j in range(len(tokenized_reviews)):
                if j == i:
                    continue
                other_review = tokenized_reviews[j]
                for k, sentence in enumerate(other_review):
                    if sentence == ' ' or sentence == '\t':
                        continue
                    row = ''
                    row += sentence.replace('\n', '[line_break_token]') + '\t'
                    row += 'O' + '\t'
                    row += 'O' + '\t'
                    row += 'Reply' + '\t'
                    row += str(review_pair_id)
                    if k != len(other_review)-1 or j != len(tokenized_reviews)-2 or i != len(tokenized_reviews)-1:
                        row += '\n'
                    output_file.write(row)
            if i != len(tokenized_reviews)-1:
                output_file.write('\n')
            review_pair_id += 1

# map (source_review_id, target_review_id, target_sentence id) -> B/I/O
# map (source_review_id, target_review_id, target_sentence id) -> [pair_labels]
def extract_mappings_from_results(result_path, n_reviews, review_lengths):
    label_mapping = {}
    pair_mapping = {}
    with open(result_path, 'r') as f:
        all_reviews = f.read().split('\n\n')
    current_entry_idx = 0
    for i in range(n_reviews):
        for j in ([i] + [x for x in range(n_reviews) if x != i]):
            for k, entry in enumerate(all_reviews[current_entry_idx].split('\n')):
                if entry == '':
                    continue
                entry_data = entry.split('\t')
                preimage = (i, j, k)
                label_mapping[str(preimage)] = entry_data[3].split('-')[0]
                raw_pairs = ast.literal_eval(entry_data[5])
                raw_head = raw_pairs[:review_lengths[i]]
                raw_pairs = raw_pairs[review_lengths[i]:]
                bound_idx = sum([x for x in review_lengths[:i]])
                raw_pairs[bound_idx:bound_idx] = raw_head
                pair_mapping[str(preimage)] = raw_pairs
        current_entry_idx += 1
    return label_mapping, pair_mapping

# map (source_review_id, target_review_id, target_sentence id) -> [pair_labels]
def extract_mappings_from_sentence_embeddings(tokenized_reviews, thresh=0.6):
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    embeddings = {} # map (review_id, sent_id) -> embedding
    for i, review in enumerate(tokenized_reviews):
        sentence_embeddings = model.encode(review)
        for j, e in enumerate(sentence_embeddings):
            embeddings[(i, j)] = e

    mapping = {}
    for ((i,j), e) in embeddings.items():
        #print([distance.cosine(e, ep) for _,ep in sorted(embeddings.items())])
        labels = [1 if distance.cosine(e, ep) < thresh else 0 for _,ep in sorted(embeddings.items())]
        #(review_match, sentence_match) = min([(distance.cosine(e, ep), (x,y)) for (x,y),ep in embeddings.items() if (x,y) != (i,j)])[1]
        for k in range(len(tokenized_reviews)):
            mapping[str((k, i, j))] = labels

    return mapping

def generate_entailment_results(tokenized_reviews):
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz", "textual_entailment")
    mapping = {}
    for i1, review1 in enumerate(tokenized_reviews):
        for j1, sentence1 in enumerate(review1):
            labels = []
            for i2, review2 in enumerate(tokenized_reviews):
                for j2, sentence2 in enumerate(review2):
                    result = predictor.predict(
                      hypothesis=sentence1,
                      premise=sentence2
                    )
                    if result['label'] == 'entailment':
                        labels.append(1)
                    else:
                        labels.append(0)
            mapping[(i1, i1, j1)] = labels
            f = open("pair_extraction/results/entailment_results.txt","w")
            f.write(str(mapping))
            f.close()

def extract_mappings_from_entail():
    with open("pair_extraction/results/entailment_results.old.txt","r") as f:
        raw_mapping = ast.literal_eval(f.read())
    return {str(x):y for x,y in raw_mapping.items()}

def generate_ner_results(tokenized_reviews, thresh=0.9):
    ner = pipeline("ner")
    entity_to_sent = defaultdict(lambda: set())
    sent_to_entity = defaultdict(lambda: set())
    for i, review in enumerate(tokenized_reviews):
        for j, sentence in enumerate(review):
            result = ner(sentence)
            entities = {r['word'] for r in result if r['score']>thresh}
            sent_to_entity[(i, j)] = sent_to_entity[(i, j)].union(entities)
            for ent in entities:
                entity_to_sent[ent].add((i, j))
    mapping = {}
    for i1, review1 in enumerate(tokenized_reviews):
        for j1, sentence1 in enumerate(review1):
            labels = []
            for i2, review2 in enumerate(tokenized_reviews):
                for j2, sentence2 in enumerate(review2):
                    if len(sent_to_entity[(i1, j1)].intersection(sent_to_entity[(i2, j2)])) > 0:
                        labels.append(1)
                    else:
                        labels.append(0)
            mapping[(i1, i1, j1)] = labels
            f = open("pair_extraction/results/ner_results.txt","w")
            f.write(str(mapping))
            f.close()
            print(labels)

def extract_mappings_from_ner():
    with open("pair_extraction/results/ner_results.txt","r") as f:
        raw_mapping = ast.literal_eval(f.read())
    return {str(x):y for x,y in raw_mapping.items()}
