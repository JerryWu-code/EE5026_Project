from libsvm.svmutil import svm_train, svm_predict
from data_loader import get_dataset
import numpy as np
from tqdm import tqdm

def perform_grid_search(X_train, y_train, X_val, y_val, param_grid):
    """
    Perform grid search for hyperparameter tuning of SVM model.
    :param X_train: Training data features
    :param y_train: Training data labels
    :param X_val: Validation data features
    :param y_val: Validation data labels
    :param param_grid: Dictionary containing hyperparameters and their respective ranges
    :return: Best hyperparameters and corresponding accuracy
    """
    best_accuracy = 0
    best_param = {}
    total_iterations = len(param_grid['C']) * len(param_grid['gamma'])

    with tqdm(total=total_iterations, desc="Grid Search Progress") as pbar:
        for C in param_grid['C']:
            for gamma in param_grid['gamma']:
                # Train the model
                param_str = '-c {} -g {} -q'.format(C, gamma)
                model = svm_train(y_train, X_train, param_str)

                # Validate the model
                p_label, p_acc, p_val = svm_predict(y_val, X_val, model, '-q')
                accuracy = p_acc[0]

                # Update the best parameters
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_param = {'C': C, 'gamma': gamma}

                # Update the progress bar
                pbar.update(1)

    return best_param, best_accuracy

if __name__ == "__main__":
    # 1. Get data
    X_train, y_train, X_test, y_test = get_dataset(train_num=None)

    # 2. Define hyperparameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001]
    }

    # 3. Perform grid search
    best_param, best_accuracy = perform_grid_search(X_train.tolist(), y_train.tolist(), X_test.tolist(), y_test.tolist(), param_grid)
    print("Best parameters found:", best_param)
    print("Best accuracy found:", best_accuracy)

    # 4. Train model with best parameters
    param_str = '-c {} -g {} -q'.format(best_param['C'], best_param['gamma'])
    model = svm_train(y_train.tolist(), X_train.tolist(), param_str)

    # 5. Predict on the test set
    p_label, p_acc, p_val = svm_predict(y_test.tolist(), X_test.tolist(), model, '-q')

    # p_acc is a tuple containing accuracy
    accuracy = p_acc[0]
    print("Test Accuracy with best parameters: {}%".format(accuracy))
