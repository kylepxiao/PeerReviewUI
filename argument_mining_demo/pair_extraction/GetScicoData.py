import openreview
import urllib.parse as urlparse
import pandas as pd
import os
import sys
import json
import spacy
import ast
from nltk import sent_tokenize, word_tokenize
from urllib.parse import parse_qs

with open('entities_to_keep.jsonl', 'r') as f:
    ents_to_keep = ast.literal_eval(f.read())

print(ents_to_keep)

client = openreview.Client(baseurl='https://api.openreview.net')
review_url = 'https://openreview.net/forum?id=ohdw3t-8VCY'
parsed = urlparse.urlparse(review_url)
forum_id = parse_qs(parsed.query)['id'][0]

public_comments = client.get_notes(forum=forum_id)

all_reviews = []

for comment in public_comments:
    if 'review' in comment.content.keys():
        all_reviews.append(comment.content['review'])

openreview_dict = {
    "id": 0,
    "source": "pwc",
    "hard": False,
    "hard_10": False,
    "hard_20": False,
    "curated": False
}

# use spacy for NER mentions
nlp = spacy.load("en_core_sci_scibert")

# variables for constructing json
flatten_tokens = []
flatten_mentions = []
tokens = []
doc_ids = []
metadata = []
sentences = []
mentions = []
relations = []
matched_ents = set()

#reviews_by_sentence = []
tokenized_reviews = []
all_word_counter = 0
for i, review in enumerate(all_reviews):
    doc = nlp(review)
    ents = {tuple(word_tokenize(str(e))) for e in doc.ents}
    tokenized_sentences = sent_tokenize(review)
    current_sentence_bounds = []
    current_tokenized_words = []
    start_bound = 0
    word_counter = 0
    for j, sentence in enumerate(tokenized_sentences):
        tokenized_words = word_tokenize(sentence)
        end_bound = start_bound + len(tokenized_words)
        flatten_tokens += tokenized_words
        current_tokenized_words += tokenized_words
        current_sentence_bounds.append([start_bound, end_bound])
        start_bound = end_bound
        for k, word in enumerate(tokenized_words):
            """mentions.append([i, word_counter, word_counter+1, all_word_counter])
            flatten_mentions.append([all_word_counter, all_word_counter+1, all_word_counter])"""
            for ent in ents:
                if tuple(tokenized_words[k:k+len(ent)]) == ent and ent in ents_to_keep:
                    matched_ents.add(ent)
                    mentions.append([i, word_counter, word_counter+len(ent), all_word_counter])
                    flatten_mentions.append([all_word_counter, all_word_counter+len(ent), all_word_counter])
            word_counter += 1
            all_word_counter += 1

        """reviews_by_sentence.append(
            {
                'review_id': i,
                'sentence_id': j,
                'sentence': sentence
            }
        )"""
    tokenized_reviews.append(tokenized_sentences)
    sentences.append(current_sentence_bounds)
    tokens.append(current_tokenized_words)
    doc_ids.append(i)

openreview_dict['flatten_tokens'] = flatten_tokens
openreview_dict['flatten_mentions'] = flatten_mentions
openreview_dict['tokens'] = tokens
openreview_dict['doc_ids'] = doc_ids
openreview_dict['metadata'] = metadata
openreview_dict['sentences'] = sentences
openreview_dict['mentions'] = mentions
openreview_dict['relations'] = relations

with open('openreview.jsonl', 'w') as f:
    json.dump(openreview_dict, f)

with open('matched_ents.jsonl', 'w') as f:
    f.write(str(matched_ents))
