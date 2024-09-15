from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

def classification_metrics(result_df):
    """
    Print classification metrics for the given result dataframe.
     Args:
        result_df (pd.DataFrame): A dataframe containing 'true_label' and 'pred_label' columns.
    """
    y_true = result_df['true_label']
    y_pred = result_df['pred_label']

    conf_matrix = confusion_matrix(y_true, y_pred, labels=['HGSC', 'CCOC', 'LGSC', 'ECOC'])
    print("Confusion Matrix:\n", conf_matrix)

    class_report = classification_report(y_true, y_pred, target_names=['HGSC', 'CCOC', 'LGSC', 'ECOC'], digits=4)
    print("Classification Report:\n", class_report)

    accuracy = accuracy_score(y_true, y_pred)
    print("Overall Accuracy:", accuracy)