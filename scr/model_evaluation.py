import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    labels = sorted(set(y_test))
    cm = confusion_matrix(y_test, predictions)
    cm_df = pd.DataFrame(cm,
                         index=[f'Actual {label}' for label in labels],
                         columns=[f'Predicted {label}' for label in labels])

    report_dict = classification_report(y_test, predictions, output_dict=True)
    report_str = classification_report(y_test, predictions)

    print(f'\nConfusion Matrix:\n{cm_df}\n')
    print(f'Classification Report:\n{report_str}\n')

    return cm_df, report_dict