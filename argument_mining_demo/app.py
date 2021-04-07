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
from pair_extraction.dataExtraction import extract_to_file, extract_mappings_from_results
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

        # prepare data for argument pair extraction
        extract_to_file(tokenized_reviews, os.path.join('pair_extraction', 'data'))
        sep_data()
        pair_inference("pair_extraction/data/predict.txt")
        pair_mapping = extract_mappings_from_results("pair_extraction/results/english_model_glove.predict.results", len(all_reviews))

        # inference for argument classification
        df_sentences = pd.DataFrame(reviews_by_sentence)
        df_arg_predicted = predict_stance_from_df(arg_model_dir, df_sentences).copy()
        df_valence_predicted = predict_stance_from_df(valence_model_dir, df_sentences).copy()

        class_data = {
            'sentences': df_arg_predicted['sentence'].tolist(),
            'review_id': df_arg_predicted['review_id'].tolist(),
            'sentence_id': df_arg_predicted['sentence_id'].tolist(),
            'arg_pred': df_arg_predicted['confidence'].tolist(),
            'valence_pred': df_valence_predicted['confidence'].tolist(),
        }
        reviews = [{'id': 'review_' + str(i), 'review': review} for i,review in enumerate(all_reviews)]
        review_ids = [x for x in range(len(all_reviews))]
        return render_template('display.html', reviews=reviews, class_data=class_data, pair_mapping=pair_mapping)

    return 'Bad Request'
