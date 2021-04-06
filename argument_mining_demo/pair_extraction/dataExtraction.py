import os
import pandas as pd

"""def extract_to_file(tokenized_reviews, output_path):
    df_ape = []
    review_pair_id = 0
    for i, review in enumerate(tokenized_reviews):
        # Get dummy labels for all sentences in a review
        for sentence in review:
            row = {}
            row['sentence'] = sentence.replace('\n', '[line_break_token]')
            row['label1'] = 'O'
            row['label2'] = 'O'
            row['type'] = 'Review'
            row['id'] = review_pair_id
            df_ape.append(row)
        # Get all combinations of reviews with other reviews
        for j in range(i+1, len(tokenized_reviews), 1):
            other_review = tokenized_reviews[j]
            for sentence in other_review:
                row = {}
                row['sentence'] = sentence.replace('\n', '[line_break_token]')
                row['label1'] = 'O'
                row['label2'] = 'O'
                row['type'] = 'Reply'
                row['id'] = review_pair_id
                df_ape.append(row)
        review_pair_id += 1
    # save to file
    df_ape = pd.DataFrame(df_ape, columns=['sentence', 'label1', 'label2', 'type', 'id'])
    df_ape.to_csv(os.path.join(output_path, 'ReviewRebuttalPredict.txt'), sep='\t', header=False, index=False)"""

def extract_to_file(tokenized_reviews, output_path):
    review_pair_id = 0
    with open(os.path.join(output_path, 'ReviewRebuttalPredict.txt'), 'w') as output_file:
        for i in range(len(tokenized_reviews)-1):
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
            output_file.write('\n')
            # Get all combinations of reviews with other reviews
            for j in range(i+1, len(tokenized_reviews), 1):
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
                    if k != len(other_review)-1 or j != len(tokenized_reviews)-1 or i != len(tokenized_reviews)-2:
                        row += '\n'
                    output_file.write(row)
                if j != len(tokenized_reviews)-1 or i != len(tokenized_reviews)-2:
                    output_file.write('\n')
            review_pair_id += 1
