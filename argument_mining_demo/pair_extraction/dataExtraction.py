import os
import ast
import pandas as pd
import numpy as np
import allennlp_models.pair_classification
import spacy
#import neuralcoref
import time
import torch
import json
from copy import deepcopy
from allennlp.predictors.predictor import Predictor
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from collections import defaultdict
from nltk import word_tokenize
from bert_score import score as bert_score

from pair_extraction.model.seq2seq import Seq2SeqModel

def abs_cosine_distance(X, Y):
    return 1 - abs(distance.cosine(X, Y) - 1)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

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
def extract_mappings_from_sentence_embeddings(tokenized_reviews, thresh=0.6, topk=None, model_name='paraphrase-distilroberta-base-v1'):
    #model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    #model = SentenceTransformer('allenai/scibert_scivocab_uncased')
    #model = SentenceTransformer('allenai/specter')
    model = SentenceTransformer(model_name)

    embeddings = {} # map (review_id, sent_id) -> embedding
    for i, review in enumerate(tokenized_reviews):
        sentence_embeddings = model.encode(review)
        for j, e in enumerate(sentence_embeddings):
            embeddings[(i, j)] = e

    mapping = {}
    for ((i,j), e) in embeddings.items():
        #print([abs_cosine_distance(e, ep) for _,ep in sorted(embeddings.items())])
        if topk is None:
            labels = [1 if abs_cosine_distance(e, ep) < thresh else 0 for _,ep in sorted(embeddings.items())]
        else:
            scores = sorted([(abs_cosine_distance(e, ep[1]),i) for i,ep in enumerate(sorted(embeddings.items()))])
            labels = [0] * len(scores)
            for score in scores[:topk]:
                labels[score[1]] = 1
        #(review_match, sentence_match) = min([(abs_cosine_distance(e, ep), (x,y)) for (x,y),ep in embeddings.items() if (x,y) != (i,j)])[1]
        for k in range(len(tokenized_reviews)):
            mapping[str((k, i, j))] = labels

    return mapping

def map_word_to_sentence(sentences, word_idx):
    sentence = 0
    while word_idx >= 0:
        word_idx -= len(word_tokenize(sentences[sentence]))
        sentence += 1
    return sentence - 1

def extract_scico_results(tokenized_reviews):
    with open('pair_extraction/results/scico_ctrlsum.jsonl') as f:
        data = dict(json.loads(f.read()))

    sentence_level_data = []
    sentence_to_cluster = defaultdict(lambda: set())
    cluster_to_sentences = defaultdict(lambda: set())
    sentence_to_keywords = defaultdict(lambda: set())
    for mention in data['mentions']:
        review = tokenized_reviews[mention[0]]
        left = map_word_to_sentence(review, mention[1])
        right = map_word_to_sentence(review, mention[2])
        for j in range(left, right+1, 1):
            sentence_to_cluster[mention[0], j].add(mention[3])
            cluster_to_sentences[mention[3]].add((mention[0], j))
            for k in range(mention[1], mention[2]+1, 1):
                sentence_to_keywords[mention[0], j].add(data['tokens'][mention[0]][k])
            #sentence_level_data.append([mention[0], map_word_to_sentence(review, mention[1]), map_word_to_sentence(review, mention[2]), mention[3]])

    pair_mapping = {}
    label_mapping = {}
    for i1, review1 in enumerate(tokenized_reviews):
        for j1, sentence1 in enumerate(review):
            current_cluster = sentence_to_cluster[i1, j1]
            labels = []
            for i2, review2 in enumerate(tokenized_reviews):
                for j2, sentence2 in enumerate(review2):
                    if len(sentence_to_cluster[i2, j2].intersection(current_cluster)) > 0:
                        labels.append(1)
                    else:
                        labels.append(0)
            pair_mapping[str((i1, i1, j1))] = labels
            if len(sentence_to_keywords[i1, j1]) > 0:
                label_mapping[str((i1, i1, j1))] = ','.join(list(sentence_to_keywords[i1, j1]))
            else:
                label_mapping[str((i1, i1, j1))] = '-'
    return label_mapping, pair_mapping



def generate_bertscore_results(tokenized_reviews, topk=6):
    flattened_sentences = [item for sublist in tokenized_reviews for item in sublist]

    embeddings = {} # map (review_id, sent_id) -> embedding
    mapping = {}
    for i1, review1 in enumerate(tokenized_reviews):
        for j1, sentence1 in enumerate(review1):
            cand = [sentence1] * len(flattened_sentences)
            _, _, scores1 = bert_score(cand, flattened_sentences, lang='en')
            _, _, scores2 = bert_score(flattened_sentences, cand, lang='en')
            scores = [(a+b)/2 for (a,b) in zip(scores1, scores2)]
            scores = sorted([(s, i) for i,s in enumerate(scores)], reverse=True)
            labels = [0] * len(scores)
            for score in scores[:topk]:
                labels[score[1]] = 1
            for k in range(len(tokenized_reviews)):
                mapping[str((k, i1, j1))] = labels
            f = open("pair_extraction/results/bertscore_results.txt","w")
            f.write(str(mapping))
            f.close()

    return mapping

def extract_bertscore_mappings():
    with open("pair_extraction/results/bertscore_results.txt", 'r') as f:
        mappings = ast.literal_eval(f.read())
    return mappings

def extract_mappings_from_finetuned_embeddings(tokenized_reviews, thresh=0.6, topk=None, model_name='pair_extraction/model_files/parasci_bart_outputs'):
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    model.load_state_dict(torch.load('pair_extraction/model_files/parasci_siamese.pt', map_location='cpu'))
    embeddings = {} # map (review_id, sent_id) -> embedding
    for i, review in enumerate(tokenized_reviews):
        sentence_embeddings = model.encode(review)
        for j, e in enumerate(sentence_embeddings):
            embeddings[(i, j)] = e

    mapping = {}
    for ((i,j), e) in embeddings.items():
        #print([abs_cosine_distance(e, ep) for _,ep in sorted(embeddings.items())])
        if topk is None:
            labels = [1 if abs_cosine_distance(e, ep) < thresh else 0 for _,ep in sorted(embeddings.items())]
        else:
            scores = sorted([(abs_cosine_distance(e, ep[1]),i) for i,ep in enumerate(sorted(embeddings.items()))])
            labels = [0] * len(scores)
            for score in scores[:topk]:
                labels[score[1]] = 1
        #(review_match, sentence_match) = min([(abs_cosine_distance(e, ep), (x,y)) for (x,y),ep in embeddings.items() if (x,y) != (i,j)])[1]
        for k in range(len(tokenized_reviews)):
            mapping[str((k, i, j))] = labels

    return mapping
    """model = Seq2SeqModel(encoder_decoder_type="bart", encoder_decoder_name=model_name)

    embeddings = {} # map (review_id, sent_id) -> embedding
    for i, review in enumerate(tokenized_reviews):
        encoded_input = model.encoder_tokenizer(review, truncation=True, padding="longest", return_tensors="pt")
        model.model = model.model.to(model.device)
        with torch.no_grad():
            outputs = model.model(encoded_input["input_ids"].to(model.device))
        sentence_embeddings = mean_pooling(outputs, encoded_input['attention_mask'].to(model.device)).cpu()
        for j, e in enumerate(sentence_embeddings):
            embeddings[(i, j)] = np.array(e)

    mapping = {}
    for ((i,j), e) in embeddings.items():
        #print([abs_cosine_distance(e, ep) for _,ep in sorted(embeddings.items())])
        if topk is None:
            labels = [1 if abs_cosine_distance(e, ep) < thresh else 0 for _,ep in sorted(embeddings.items())]
        else:
            scores = sorted([(abs_cosine_distance(e, ep[1]),i) for i,ep in enumerate(sorted(embeddings.items()))])
            labels = [0] * len(scores)
            for score in scores[:topk]:
                labels[score[1]] = 1
        #(review_match, sentence_match) = min([(abs_cosine_distance(e, ep), (x,y)) for (x,y),ep in embeddings.items() if (x,y) != (i,j)])[1]
        for k in range(len(tokenized_reviews)):
            mapping[str((k, i, j))] = labels

    return mapping"""

def generate_entailment_results(tokenized_reviews):
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/decomposable-attention-elmo-2020.04.09.tar.gz", "textual_entailment")
    pair_mapping = {}
    total = sum([len(sentences) for sentences in tokenized_reviews])
    for i, review in enumerate(tokenized_reviews):
        for j, sentence in enumerate(review):
            if str((i, i, j)) not in pair_mapping.keys():
                pair_mapping[str((i, i, j))] = [0] * total
                if i > 0:
                    idx = sum([len(review) for review in tokenized_reviews[:i]]) + j
                else:
                    idx = j
                pair_mapping[str((i, i, j))][idx] = 1
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
            pair_mapping[(i1, i1, j1)] = labels
            f = open("pair_extraction/results/entailment_results.txt","w")
            f.write(str(pair_mapping))
            f.close()

def extract_mappings_from_entail(tokenized_reviews):
    with open("pair_extraction/results/entailment_results.txt","r") as f:
        raw_mapping = ast.literal_eval(f.read())
    pair_mapping = {str(x):y for x,y in raw_mapping.items()}
    total = sum([len(sentences) for sentences in tokenized_reviews])
    for i, review in enumerate(tokenized_reviews):
        for j, sentence in enumerate(review):
            if str((i, i, j)) not in pair_mapping.keys():
                print(i, i, j, "not in entailment mapping")
            pair_mapping[str((i, i, j))] = [0]*total
            if i > 0:
                idx = sum([len(review) for review in tokenized_reviews[:i]]) + j
            else:
                idx = j
            pair_mapping[str((i, i, j))][idx] = 1
    return {str(x):y for x,y in raw_mapping.items()}

def generate_ner_results(tokenized_reviews, thresh=0.8):
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
    pair_mapping = {}
    label_mapping = {}
    for i1, review1 in enumerate(tokenized_reviews):
        for j1, sentence1 in enumerate(review1):
            labels = []
            all_common = set()
            for i2, review2 in enumerate(tokenized_reviews):
                for j2, sentence2 in enumerate(review2):
                    common = sent_to_entity[(i1, j1)].intersection(sent_to_entity[(i2, j2)])
                    all_common = all_common.union(common)
                    if len(common) > 0:
                        labels.append(1)
                    else:
                        labels.append(0)
            label_mapping[str((i1, i1, j1))] = str(all_common) if len(all_common) > 0 else '-'
            pair_mapping[(i1, i1, j1)] = labels
            f = open("pair_extraction/results/ner_pair_mapping.txt","w")
            f.write(str(pair_mapping))
            f.close()
            f = open("pair_extraction/results/ner_label_mapping.txt","w")
            f.write(str(label_mapping))
            f.close()

def extract_mappings_from_ner():
    with open("pair_extraction/results/ner_pair_mapping.txt","r") as f:
        raw_mapping = ast.literal_eval(f.read())
        pair_mapping = {str(x):y for x,y in raw_mapping.items()}
    with open("pair_extraction/results/ner_label_mapping.txt","r") as f:
        raw_mapping = ast.literal_eval(f.read())
        label_mapping = {str(x):y for x,y in raw_mapping.items()}
    return label_mapping, pair_mapping

def get_ids_from_span_idx(review_lengths, idx):
    review_id = 0
    while idx >= 0:
        for sentence_lengths in review_lengths:
            sentence_id = 0
            for x in sentence_lengths:
                if x >= idx+1:
                    return (review_id, sentence_id)
                else:
                    sentence_id += 1
                    idx -= x
            review_id += 1
    return (review_id, sentence_id)

def extract_mappings_from_coreference(tokenized_reviews):
    try:
        nlp = spacy.load('en')
        neuralcoref.add_to_pipe(nlp)
        flattened_reviews = []
        review_lengths = []
        for review in tokenized_reviews:
            flattened_reviews += review
            sentence_lengths = []
            for sentence in review:
                sentence_lengths.append(len(nlp(sentence)))
            review_lengths.append(sentence_lengths)
        flattened_reviews = ' '.join(flattened_reviews)
        doc = nlp(flattened_reviews)

        # construct mapping
        pair_mapping = {}
        label_mapping = {}
        for cluster in doc._.coref_clusters:
            pairs_from_clusters = []
            for mention in cluster.mentions:
                review_id, sentence_id = get_ids_from_span_idx(review_lengths, mention.start)
                pairs_from_clusters.append((review_id, sentence_id))
                label_mapping[str((review_id, review_id, sentence_id))] = str(mention)
            for a in pairs_from_clusters:
                labels = []
                for i, review in enumerate(tokenized_reviews):
                    for j, sentence in enumerate(review):
                        if (i,j) in pairs_from_clusters:
                            labels.append(1)
                        else:
                            labels.append(0)
                        if str((i, i, j)) not in label_mapping.keys():
                            label_mapping[str((i, i, j))] = '-'
                pair_mapping[str((a[0], a[0], a[1]))] = labels
        total = sum([len(sentences) for sentences in tokenized_reviews])
        for i, review in enumerate(tokenized_reviews):
            for j, sentence in enumerate(review):
                if str((i, i, j)) not in pair_mapping.keys():
                    pair_mapping[str((i, i, j))] = [0] * total
                    if i > 0:
                        idx = sum([len(review) for review in tokenized_reviews[:i]]) + j
                    else:
                        idx = j
                    pair_mapping[str((i, i, j))][idx] = 1
    except:
        print("Error with neural coreference!")
        pair_mapping = {}
        label_mapping = {}
        total = sum([len(review) for review in tokenized_reviews])
        for i, review in enumerate(tokenized_reviews):
            for j, sentence in enumerate(review):
                pair_mapping[str((i, i, j))] = [0] * total
                label_mapping[str((i, i, j))] = '-'


    return label_mapping, pair_mapping


def extract_groundtruth_mapping(tokenized_reviews):
    pair_mapping = {}
    with open("pair_extraction/results/ground_truth_raw.txt", 'r') as f:
        for line in f.readlines():
            pairs = line.replace(' ', '').replace('\n', '').split(';')
            pairs = {tuple([int(p) for p in x.split(',')]) for x in pairs}
            for a in pairs:
                labels = []
                for i, review in enumerate(tokenized_reviews):
                    for j, sentence in enumerate(review):
                        if (i,j) in pairs or (a[0], a[1]) == (i,j):
                            labels.append(1)
                        else:
                            labels.append(0)
                pair_mapping[str((a[0], a[0], a[1]))] = labels
    total = sum([len(review) for review in tokenized_reviews])
    for i, review in enumerate(tokenized_reviews):
        for j, sentence in enumerate(review):
            if str((i, i, j)) not in pair_mapping.keys():
                pair_mapping[str((i, i, j))] = [0] * total
                if i > 0:
                    idx = sum([len(review) for review in tokenized_reviews[:i]]) + j
                else:
                    idx = j
                pair_mapping[str((i, i, j))][idx] = 1
    return pair_mapping

def prep_scico_data(tokenized_reviews):
    final_dict = {}
    final_dict['flatten_tokens'] = []
    final_dict['flatten_mentions'] = []
    final_dict['tokens'] = []
    final_dict['doc_ids'] = []
    final_dict['metadata'] = []
    final_dict['sentences'] = []
    final_dict['mentions'] = []
    final_dict['relations'] = []
    final_dict['id'] = 0
    final_dict['hard_10'] = False
    final_dict['hard_20'] = False
    final_dict["source"] = "pwc"
    for i, review in enumerate(tokenized_reviews):
        for j, sentence in enumerate(review):
            tokenized_sentence = word_tokenize(sentence)
            for word in tokenized_sentence:
                final_dict['flatten_tokens'].append(word)
    flattened_tokens = []

def extract_groundtruth_from_df(df):
    pass
    """for paper_id in df['paper_id'].unique():
        pair_mapping = {}
        reviews = df[df['paper_id'] == paper_id]
        for i, row in df.iterrows():
            row[]"""
