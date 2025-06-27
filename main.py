from scr.preprocessing import load_and_process_data
from scr.model_training import split_and_scale, train_model
from scr.model_evaluation import evaluate_model

X, y = load_and_process_data('onlinefraud.csv')
X_train, X_test, y_train, y_test = split_and_scale(X, y)
model = train_model(X_train, y_train)
cm_df, report_dict = evaluate_model(model, X_test, y_test)