import os
import ast
import pandas as pd

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
