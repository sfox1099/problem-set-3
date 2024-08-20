'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists

    '''

    # Initializes variables for micro calculations
    total_tp = 0  # True Positives for all genres
    total_fp = 0  # False Positives for all genres
    total_fn = 0  # False Negatives for all genres
    
    # Initializes lists for macro metrics
    macro_prec_list = []
    macro_recall_list = []
    macro_f1_list = []

    # Loops through each genre to calculate precision, recall, and F1
    for genre in genre_list:
        tp = genre_tp_counts[genre]
        fp = genre_fp_counts[genre]
        fn = genre_true_counts[genre] - tp
        
        # Precision, recall, and F1 for this genre
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Appends macro metrics
        macro_prec_list.append(precision)
        macro_recall_list.append(recall)
        macro_f1_list.append(f1)
        
        # Accumulates totals for micro metrics
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Calculates micro metrics
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    return micro_precision, micro_recall, micro_f1, macro_prec_list, macro_recall_list, macro_f1_list

    
def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrixes for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''

     # Initializes lists to hold the binary values for each genre
    pred_rows = []
    true_rows = []

    # Loops through each row in the dataframe
    for i, row in model_pred_df.iterrows():
        # Initializes binary vectors for the current movie
        pred_vector = [0] * len(genre_list)
        true_vector = [0] * len(genre_list)
        
        # Sets 1 for the predicted genre in pred_vector
        if row['predicted'] in genre_list:
            pred_vector[genre_list.index(row['predicted'])] = 1
        
        # Sets 1 for the actual genres in true_vector
        actual_genres = eval(row['actual genres'])
        for genre in actual_genres:
            if genre in genre_list:
                true_vector[genre_list.index(genre)] = 1
        
        # Appends the binary vectors to the rows list
        pred_rows.append(pred_vector)
        true_rows.append(true_vector)
    
    # Converts the lists into pandas DataFrames (like matrices)
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    
    # Calculates precision, recall, and f1 using sklearn for both micro and macro averages
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(true_matrix, pred_matrix, average='macro', zero_division=0)
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(true_matrix, pred_matrix, average='micro', zero_division=0)
    
    return macro_prec, macro_rec, macro_f1, micro_prec, micro_rec, micro_f1
