"""Evaluating the model with Recall, Precisiona and F1 Score"""
import pandas as pd
import numpy as np



def recall_precision_f1(confusionMatrix_DF):
    '''
    Args:
        confusionMatrix_DF: a datafram with index_col=0
        
    returns: (x, y)
        x: is a datafrom of recall and precision for every class
        y: is the f1 score for the model.
        
    '''
    
    confusionMatrix_np  = confusionMatrix_DF.values
    bahr = 0
    sum_rows = []
    sum_columns = []
    diagonal_recall = []
    diagonal_precision = []
    
    matrices = [confusionMatrix_np, np.transpose(confusionMatrix_np)]
    flag = 0
    for matrix in matrices:
        # print(matrix)
        for x in matrix:
            # class_num is the sum of the ith row of the confusion matrix
            # Also it it the sum of the ith column, when flag = 1
            class_sum = np.sum(x)
            # Recall
            if flag == 0:
                sum_rows.append(class_sum)
                diagonal_recall.append(x[bahr])
            # Precision
            elif flag == 1:
                sum_columns.append(class_sum)
                diagonal_precision.append(x[bahr])
            
            bahr += 1 
        flag += 1
        bahr = 0
        
    '''
    # Recall per class
    print(np.array(diagonal_recall)/ np.array(sum_rows))
    # Precision per class
    print(np.array(diagonal_precision)/ np.array(sum_columns))
    '''
    recall_per_class = np.array(diagonal_recall)/ np.array(sum_rows)
    precision_per_class = np.array(diagonal_precision)/ np.array(sum_columns)
    
    
    recall_mean = np.mean(recall_per_class)
    precision_mean = np.mean(precision_per_class)
    
    sum_rec_pre = recall_mean + precision_mean
    mul_rec_re  = recall_mean * precision_mean
    f1_score = 2 * (mul_rec_re / sum_rec_pre)
    
    # Building the Data Frame
    resulat_dict = {'Recall': recall_per_class,
                    'Precision': precision_per_class
                   }
    resulat_DF = pd.DataFrame(resulat_dict, dtype=float)
    resulat_DF.index = confusionMatrix_DF.index
    resulat_DF
    
    return resulat_DF, f1_score


'''
# Example
df = pd.read_csv('cm_df.csv', index_col=0)
dataFrame, f1 = recall_precision_f1(df)
print(f1)
print(dataFrame)
'''
