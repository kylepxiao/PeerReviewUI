import openreview
import urllib.parse as urlparse
import pandas as pd
import os
import sys
from nltk.tokenize import sent_tokenize
from flask import Flask, render_template, request
from urllib.parse import parse_qs
from argument_mining.executables.s_predict import *
sys.path.insert(1, 'pair_extraction')
#from pair_extraction.dataExtraction import extract_to_file, extract_mappings_from_results, extract_mappings_from_sentence_embeddings
from pair_extraction.dataExtraction import *
from pair_extraction.dataProcessing import sep_data
from pair_extraction.run_model import pair_inference

import nltk
nltk.download('punkt')

# Define model
arg_model_dir = os.path.join("argument_mining", "model_arg_classification")
valence_model_dir = os.path.join("argument_mining", "model_valence_classification")
#preload_model(model_dir)

# Start app and api client
app = Flask(__name__)
client = openreview.Client(baseurl='https://api.openreview.net')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review_url = str(request.form['review_url'])
        #return str(request.form['review_url'])
        parsed = urlparse.urlparse(review_url)
        forum_id = parse_qs(parsed.query)['id'][0]

        public_comments = client.get_notes(forum=forum_id)

        all_reviews = []

        for comment in public_comments:
            if 'review' in comment.content.keys():
                all_reviews.append(comment.content['review'])

        reviews_by_sentence = []
        tokenized_reviews = []
        for i, review in enumerate(all_reviews):
            sentences = sent_tokenize(review)
            for j, sentence in enumerate(sentences):
                reviews_by_sentence.append(
                    {
                        'review_id': i,
                        'sentence_id': j,
                        'sentence': sentence
                    }
                )
            tokenized_reviews.append(sentences)

        pair_mapping_gold = extract_groundtruth_mapping(tokenized_reviews)

        label_mapping_scico, pair_mapping_scico = extract_scico_results(tokenized_reviews)

        #pair_mapping_bertscore = generate_bertscore_results(tokenized_reviews, topk=6)
        pair_mapping_bertscore = extract_bertscore_mappings()


        label_mapping_coref, pair_mapping_coref = extract_mappings_from_coreference(tokenized_reviews)

        # prepare data for argument pair extraction
        review_lengths = [len(r) for r in tokenized_reviews]
        #extract_to_file(tokenized_reviews, os.path.join('pair_extraction', 'data'))
        #sep_data()
        #pair_inference("pair_extraction/data/predict.txt")
        label_mapping_ape, pair_mapping_ape = extract_mappings_from_results("pair_extraction/results/english_model_glove.predict.results", len(all_reviews), review_lengths)
        if label_mapping_ape is not None:
            pass
            #TODO change so that tokenized_reviews is modified accordingly

        # get entailment pair mapping
        # generate_entailment_results(tokenized_reviews)
        pair_mapping_entail = extract_mappings_from_entail(tokenized_reviews)

        # get ner pair mapping
        #generate_ner_results(tokenized_reviews)
        label_mapping_ner, pair_mapping_ner = extract_mappings_from_ner()

        print("Extracting parasci embeddings...")
        pair_mapping_emb_parasci = extract_mappings_from_finetuned_embeddings(tokenized_reviews, topk=5, model_name='pair_extraction/model_files/parasci_bart_outputs')

        # get sentence embedding pair mapping
        print("Extracting specter embeddings...")
        #pair_mapping_emb_specter = extract_mappings_from_sentence_embeddings(tokenized_reviews, thresh=0.15, model_name='allenai/specter')
        pair_mapping_emb_specter = extract_mappings_from_sentence_embeddings(tokenized_reviews, topk=5, model_name='allenai/specter')
        print("Extracting scibert embeddings...")
        #pair_mapping_emb_scibert = extract_mappings_from_sentence_embeddings(tokenized_reviews, thresh=0.15, model_name='allenai/scibert_scivocab_uncased')
        #pair_mapping_emb_scibert = extract_mappings_from_sentence_embeddings(tokenized_reviews, topk=5, model_name='allenai/scibert_scivocab_uncased')
        pair_mapping_emb_scibert = extract_mappings_from_sentence_embeddings(tokenized_reviews, topk=5, model_name='gsarti/scibert-nli')
        print("Extracting bert embeddings...")
        #pair_mapping_emb_bert = extract_mappings_from_sentence_embeddings(tokenized_reviews, thresh=0.6, model_name='paraphrase-distilroberta-base-v1')
        pair_mapping_emb_bert = extract_mappings_from_sentence_embeddings(tokenized_reviews, topk=5, model_name='paraphrase-distilroberta-base-v1')
        #pair_mapping_emb_bert = extract_mappings_from_sentence_embeddings(tokenized_reviews, topk=5, model_name='paraphrase-mpnet-base-v2')

        pair_mappings = {
            'ape': pair_mapping_ape,
            'entail': pair_mapping_entail,
            'ner': pair_mapping_ner,
            'emb_bert': pair_mapping_emb_bert,
            'emb_scibert': pair_mapping_emb_scibert,
            'emb_specter': pair_mapping_emb_specter,
            'coref': pair_mapping_coref,
            'emb_parasci': pair_mapping_emb_parasci,
            'bertscore': pair_mapping_bertscore,
            'scico': pair_mapping_scico,
            'gold': pair_mapping_gold
        }

        label_mappings = {
            'ape': label_mapping_ape,
            'ner': label_mapping_ner,
            'coref': label_mapping_coref,
            'scico': label_mapping_scico
        }

        # inference for argument classification
        df_sentences = pd.DataFrame(reviews_by_sentence)
        df_arg_predicted = predict_stance_from_df(arg_model_dir, df_sentences).copy()
        df_valence_predicted = predict_stance_from_df(valence_model_dir, df_sentences).copy()

        class_data = {
            #'sentences': df_arg_predicted['sentence'].tolist(),
            'sentences': [x['sentence'] for x in reviews_by_sentence],
            #'review_id': df_arg_predicted['review_id'].tolist(),
            'review_id': [x['review_id'] for x in reviews_by_sentence],
            #'sentence_id': df_arg_predicted['sentence_id'].tolist(),
            'sentence_id': [x['sentence_id'] for x in reviews_by_sentence],
            'arg_pred': df_arg_predicted['confidence'].tolist(),
            'valence_pred': df_valence_predicted['confidence'].tolist(),
        }
        reviews = [{'id': 'review_' + str(i), 'review': review} for i,review in enumerate(all_reviews)]
        review_ids = [x for x in range(len(all_reviews))]
        #return render_template('display.html', reviews=reviews, class_data=class_data, label_mapping=label_mapping, pair_mapping_ape=pair_mapping_ape, pair_mapping_entail=pair_mapping_entail, pair_mapping_emb=pair_mapping_emb, pair_mapping_ner=pair_mapping_ner)
        return render_template('display.html', reviews=reviews, class_data=class_data, label_mappings=label_mappings, pair_mappings=pair_mappings)

    return 'Bad Request'
