import pandas as pd

df_all_reviews = pd.read_csv('all_reviews_by_sentences.csv')

df_all_reviews = df_all_reviews.drop(['rating', 'decision'], axis=1)

df_all_reviews = df_all_reviews[['sentence']]
print(df_all_reviews.columns)

df_all_reviews.to_csv('all_reviews_no_labels.csv')
