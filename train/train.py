# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import emlearn
import seaborn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

current_file_path = os.path.abspath(__file__)
here = os.path.dirname(current_file_path)


def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    return accuracy, precision, recall, f1

# %%
# Correctness checking
# ------------------------------
#
# Compare predictions made by converted emlearn C model
# with those of the trained Python model
def check_correctness(out_dir, name, model_filename, test_data, test_predictions, feature_columns):
    test_res = np.array(test_predictions).flatten()

    test_dataset = "\n".join([
        emlearn.cgen.array_declare(f"{name}_testset_data", dtype='float', values=test_data),
        emlearn.cgen.array_declare(f"{name}_testset_results", dtype='int', values=test_res),
        emlearn.cgen.constant_declare(f'{name}_testset_features', val=len(feature_columns)),
        emlearn.cgen.constant_declare(f'{name}_testset_samples', val=len(test_predictions)),
    ])

    test_code = test_dataset + \
    f'''
    #include "{model_filename}" // emlearn generated model

    #include <stdio.h> // printf

    int
    {name}_test() {{
        const int n_features = {name}_testset_features;
        const int n_testcases = {name}_testset_samples;

        int errors = 0;

        for (int i=0; i<n_testcases; i++) {{
            const float *features = {name}_testset_data + (i*n_features);
            const int expect_result = {name}_testset_results[i*1];

            const int32_t out = model_predict(features, n_features);

            if (out != expect_result) {{
                printf(\"test-fail sample=%d expect=%d got=%d \\n\", i, expect_result, out);
                errors += 1;
            }}

        }}
        return errors;
    }}

    int
    main(int argc, const char *argv[])
    {{
        const int errors = {name}_test();
        return errors;
    }}'''

    test_source_file = os.path.join(out_dir, f'test_{name}.c')
    with open(test_source_file, 'w') as f:
        f.write(test_code)

    print('Generated', test_source_file)

    include_dirs = [ emlearn.includedir ]
    test_executable = emlearn.common.compile_executable(
            test_source_file,
            out_dir,
            name='test_{name}',
            include_dirs=include_dirs
    )

    import subprocess
    errors = None
    try:
        subprocess.check_output([test_executable])
        errors = 0
    except subprocess.CalledProcessError as e:
        errors = e.returncode


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_file = os.path.join(here, "wine-quality.csv")
    
    data = pd.read_csv(csv_file, sep=",")
    
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    target_column = "quality"
    feature_columns = list(set(data.columns) - set([target_column]))
    feature_columns = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]

    # # The predicted column is "quality" which is a scalar from [3, 9]
    # train_x = train.drop(target_column, axis=1)
    # test_x = test.drop(target_column, axis=1)
    # train_y = train[target_column]
    # test_y = test[target_column]

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    with mlflow.start_run():
        name ="random_forest"
        name2 ="random_forest_cmodel"
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
        model.fit(train[feature_columns], train[target_column])

        test_pred = model.predict(test[feature_columns])

        (accuracy, precision, recall, f1) = eval_metrics(test[target_column], test_pred)


        print("RandomForestClassifier model (n_estimators={:f}:".format(n_estimators))
        print("  accuracy: %s" % accuracy)
        print("  precision: %s" % precision)
        print("  recall: %s" % recall)
        print("  f1: %s" % f1)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        
        
        mlflow.sklearn.log_model(model, name)


        out_dir = os.path.join(here, 'classifiers')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        model_filename = os.path.join(out_dir, f'{name}_model.h')
        cmodel = emlearn.convert(model)
        code = cmodel.save(file=model_filename, name='model')
        
        test_pred_2 = cmodel.predict(test[feature_columns])

        # Generate a test dataet
        test_data = np.array(test[feature_columns]).flatten()

        errors = check_correctness(out_dir, name, model_filename, test_data, test_pred_2, feature_columns)
        print(f"Tested {name}: {errors} errors")

        (accuracy_2, precision_2, recall_2, f1_2) = eval_metrics(test[target_column], test_pred_2)


        print("RandomForestClassifier cmodel (n_estimators={:f}:".format(n_estimators))
        print("  accuracy_2: %s" % accuracy_2)
        print("  precision_2: %s" % precision_2)
        print("  recall_2: %s" % recall_2)
        print("  f1_2: %s" % f1_2)

        # mlflow.log_param("n_estimators", n_estimators)
        # mlflow.log_metric("accuracy", accuracy_2)
        # mlflow.log_metric("precision", precision_2)
        # mlflow.log_metric("recall", recall_2)
        # mlflow.log_metric("f1", f1_2)
        
        
        # mlflow.sklearn.log_model(cmodel, name2)