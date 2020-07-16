import numpy as np
np.random.seed(773)
import random
random.seed(773)
import os
os.environ['PYTHONHASHSEED'] = '0'

import tensorflow as tf
config = tf.ConfigProto()
from keras import backend as K
tf.set_random_seed(773)

config.gpu_options.allow_growth=True   # Don't pre-allocate memory; allocate as-needed
config.gpu_options.allocator_type='BFC'

sess = tf.Session(graph=tf.get_default_graph(), config=config)
K.set_session(sess)

import argparse
import nltk
import numpy as np
import os
import sklearn as sk
import pandas as pd
import pickle

from keras.engine.saving import load_model
from pandas import DataFrame
from typing import List
from typing import Dict
from inspect import signature

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.externals import joblib

from utils.experiment_utils import MutationDiseaseFeatureConfigurationProvider, DrugDrugFeatureConfigurationProvider, \
    FeatureConfigurationProvider, EvaluationResult, DataSetReader, DataPreparationPipeline
from utils.experiment_utils import  MultiLayerPerceptron as MLP
from utils.log_utils import LoggingMixin


nltk.download('punkt')


class ClassificationExperiment(LoggingMixin):

    def __init__(self):
        LoggingMixin.__init__(self)

    def create_classifier_configs(self):
        return [
            ("MLP-200", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [200])),
            ("MLP-400", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [300])),
            ("MLP-600", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [500])),
            ("MLP-800", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [800])),
            ("MLP-1000", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [1000])),

            ("MLP-200-BN", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [200], bn=True)),
            ("MLP-400-BN", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [400], bn=True)),
            ("MLP-600-BN", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [600], bn=True)),
            ("MLP-800-BN", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [800], bn=True)),
            ("MLP-1000-BN", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [1000], bn=True)),

            ("MLP-200-DO", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [200], do=True)),
            ("MLP-400-DO", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [400], do=True)),
            ("MLP-600-DO", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [600], do=True)),
            ("MLP-800-DO", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [800], do=True)),
            ("MLP-1000-DO", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [1000], do=True)),

            ("MLP-200-BN-DO", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [200], bn=True, do=True)),
            ("MLP-400-BN-DO", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [400], bn=True, do=True)),
            ("MLP-600-BN-DO", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [600], bn=True, do=True)),
            ("MLP-800-BN-DO", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [800], bn=True, do=True)),
            ("MLP-1000-BN-DO", lambda X_val, y_val: MLP.create_mlp_model(X_val, y_val, [1000], bn=True, do=True))
        ]

    def run_evaluation(self, data_set_name: str, data_set: DataFrame, document_store: Dict,
                       feature_provider: FeatureConfigurationProvider, activated_features: List[str],
                       result_directory: str) -> None:
        self.log_info("Start evaluation with data set %s", data_set_name)
        os.makedirs(result_directory, exist_ok=True)

        self.log_info("Splitting data set to dev, val and test data set")
        train_data_set, val_data_set, test_data_set = self.split_into_dev_val_and_test(data_set, ["label"], test_ratio=0.30, val_ratio=0.25)
        self.save_splits(train_data_set, val_data_set, test_data_set, result_directory)

        self.log_info("Split: %s train, %s validation, %s test", len(train_data_set), len(val_data_set), len(test_data_set))

        self.log_info(f"Evaluate features {activated_features} with {feature_provider.__class__.__name__}")
        feature_configs = feature_provider.create_feature_configurations(activated_features)
        self.log_info(f"Created feature configurations. Found {len(feature_configs)} in total")

        classifier_configs = self.create_classifier_configs()
        experiments = [(feature, classifier) for feature in feature_configs for classifier in classifier_configs]

        self.log_info("Start performing %s experiments (%s feature configs and %s classifiers)",
                    len(experiments), len(feature_configs), len(classifier_configs))

        train_data = train_label = val_data = val_label = test_data = test_label = None
        prev_feature = ""

        pp_pipeline = DataPreparationPipeline()

        for (i, (feature_config, classifier_config)) in enumerate(experiments):
            feature_name = feature_config[1]
            configuration = feature_config[2]

            classifier_name = classifier_config[0]
            classifier_factory = classifier_config[1]
            experiment_name = feature_name + "-" + classifier_name

            self.log_info("Starting experiment %s (%s / %s)", experiment_name, (i + 1), len(experiments))

            if feature_config != prev_feature:
                data_pipeline = pp_pipeline.get_data_pipeline(feature_name, configuration, document_store)
                label_pipeline = pp_pipeline.get_label_pipeline("label")

                self.log_info("Start preparation of train data set")
                train_data, train_label = data_pipeline.fit_transform(train_data_set), label_pipeline.fit_transform(train_data_set)

                self.log_info("Start preparation of validation data set")
                val_data, val_label = data_pipeline.transform(val_data_set), label_pipeline.transform(val_data_set)

                self.log_info("Start preparation of test data set")
                test_data, test_label = data_pipeline.transform(test_data_set), label_pipeline.transform(test_data_set)

            else:
                self.log_info("Reuse prepared data from last iteration!")

            prev_feature = feature_config

            self.log_info("Start training of classifier %s (%s)", classifier_name, str(train_data.shape))

            factory_signature = signature(classifier_factory)
            num_factory_parameters = len(factory_signature.parameters)
            if num_factory_parameters == 0:
                all_train_data = np.concatenate((train_data, val_data), axis=0)
                all_train_label = np.concatenate((train_label, val_label), axis=0)

                classifier = classifier_factory()
                classifier.fit(all_train_data, all_train_label)

            elif num_factory_parameters == 2:
                classifier = classifier_factory(val_data, val_label)
                classifier.fit(train_data, train_label)

            else:
                raise Exception("Found classifier factory with unsupported number of parameters: %s", num_factory_parameters)

            self.log_info("Start evaluation of trained classifier")

            train_result = self.predict_and_evaluate(classifier, train_data, train_label)
            val_result = self.predict_and_evaluate(classifier, val_data, val_label)
            test_result = self.predict_and_evaluate(classifier, test_data, test_label)

            self.log_info("Finished evaluation: train-acc=%0.4f | train-f1=%0.4f | val-acc=%0.4f | val-f1=%0.4f | "
                          "test-acc=%0.4f | test-f1 = %0.4f", train_result.accuracy_score, train_result.f1_score,
                          val_result.accuracy_score, val_result.f1_score, test_result.accuracy_score,
                          test_result.f1_score)

            self.append_result(feature_config[0], classifier_name, train_result, val_result,
                               test_result, result_directory)

            experiment_directory = os.path.join(result_directory, experiment_name)
            os.makedirs(experiment_directory, exist_ok=True)

            model_file = os.path.join(experiment_directory, "model.cls")
            self.log_info("Saving classifier to %s", model_file)
            if not isinstance(classifier, KerasClassifier):
                joblib.dump(classifier, model_file)
            else:
                best_model = load_model("mlp_model.tmp")
                best_model.save(model_file)

                os.remove("mlp_model.tmp")

            if feature_name in ("bow", "tfidf"):
                vectorizer_file = os.path.join(result_directory, "vectorizer.pk")
                self.log_info(f"Saving vectorizer to {vectorizer_file}")

                vectorizer = configuration[f"{feature_name}_vectorizer"]
                with open(vectorizer_file, "wb") as writer:
                    pickle.dump(vectorizer, writer, protocol=pickle.HIGHEST_PROTOCOL)

            train_prediction_file = os.path.join(experiment_directory, "train_prediction.tsv")
            self.log_info("Saving train predictions to %s", train_prediction_file)
            self.save_predictions(train_data_set, train_result, train_prediction_file)

            val_prediction_file = os.path.join(experiment_directory, "val_pred.tsv")
            self.log_info("Saving val predictions to %s", val_prediction_file)
            self.save_predictions(val_data_set, val_result, val_prediction_file)

            test_prediction_file = os.path.join(experiment_directory, "test_pred.tsv")
            self.log_info("Saving test predictions to %s", test_prediction_file)
            self.save_predictions(test_data_set, test_result, test_prediction_file)

        self.log_info("Finished all experiments!")

    # ------------------------------------------------------------------------------

    def save_splits(self, train_ds: DataFrame, val_ds: DataFrame, test_ds: DataFrame, result_directory: str):
        data_sets = [("train.csv", train_ds), ("val.csv", val_ds), ("test.csv", test_ds)]
        for file_name, data_set in data_sets:
            output_ds = data_set[["source_id", "target_id", "label"]]

            output_file = os.path.join(result_directory, file_name)
            output_ds.to_csv(output_file, sep="\t", encoding="utf-8", index=False)

    def split_into_train_and_test(self, data_set: DataFrame, columns: List[str], test_ratio: float = 0.2, random_seed: int = 773):
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
        split = splitter.split(data_set, data_set[columns])
        for train_indices, test_indices in split:
            training_data = data_set.iloc[train_indices]
            test_data = data_set.iloc[test_indices]

        return training_data, test_data

    def split_into_dev_val_and_test(self, data_set: DataFrame, columns: List[str], val_ratio: float = 0.2,
                                    test_ratio: float = 0.2, random_seed: int = 773):
        train_data, test_data = self.split_into_train_and_test(data_set, columns, test_ratio, random_seed)
        dev_data, val_data = self.split_into_train_and_test(train_data, columns, val_ratio, random_seed)
        return dev_data, val_data, test_data

    def predict_and_evaluate(self, classifier, data: np.ndarray, gold_labels: np.ndarray) -> EvaluationResult:
        pred_probabilities = classifier.predict_proba(data)
        pred_labels = np.argmax(pred_probabilities, axis=1)

        accuracy = sk.metrics.accuracy_score(gold_labels, pred_labels)
        f1_score = sk.metrics.f1_score(gold_labels, pred_labels)

        return EvaluationResult(pred_probabilities, pred_labels, accuracy, f1_score)

    def save_predictions(self, data_set: DataFrame, result: EvaluationResult, output_file: str) -> None:
        instances_pd = data_set[["source_id", "target_id", "label"]]
        probabilities_pd = pd.DataFrame(result.pred_probabilities, columns=["prob0", "prob1"], index=instances_pd.index)
        prediction_pd = pd.DataFrame(result.pred_labels, columns=["prediction"], index=instances_pd.index)

        complete_pd = pd.concat([instances_pd, prediction_pd, probabilities_pd], axis=1)
        complete_pd.to_csv(output_file, sep="\t", index=False)

    def append_result(self, feature: str, classifier: str, train_result: EvaluationResult,
                      val_result: EvaluationResult, test_result: EvaluationResult, result_dir: str):

        metric_raw_values = [train_result.accuracy_score, train_result.f1_score]  + \
                            [val_result.accuracy_score, val_result.f1_score] + \
                            [test_result.accuracy_score, test_result.f1_score]
        metric_values = ["{:.5f}".format(value) for value in metric_raw_values]

        result_file = os.path.join(result_dir, "results.tsv")
        with open(result_file, "a", encoding="utf-8") as file_writer:
            file_writer.write("\t".join([feature, classifier] + metric_values) + "\n")
            file_writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Entity-pair-classification")
    parser.add_argument("dataset", choices=["md", "dd"], type=str,
                        help="Type of the data set (mutation-disease (md) or drug-drug (dd)")
    parser.add_argument("dataset_dir", type=str,
                        help="Path to the data folder of the data set")
    parser.add_argument("embedding_dir", type=str,
                        help="Path to the directory containing entity and pair embeddings for the data set")
    parser.add_argument("features", choices=["entity", "pair"], nargs="+",
                        help="Features to use / evaluate")
    parser.add_argument("result_dir", type=str,
                        help="Path to the output directory")
    arguments = parser.parse_args()

    # Create feature configuration provider
    if arguments.dataset == "md":
        feature_provider = MutationDiseaseFeatureConfigurationProvider(arguments.embedding_dir)
    elif arguments.dataset == "dd":
        feature_provider = DrugDrugFeatureConfigurationProvider(arguments.embedding_dir)
    else:
        raise NotImplementedError(f"Unsupported data set {arguments.dataset}")

    activated_features = arguments.features

    # Read positive + negative instance + documents
    ds_reader = DataSetReader(arguments.dataset_dir)
    documents = ds_reader.read_documents()
    data_set = ds_reader.read_data_set()

    classification = ClassificationExperiment()
    classification.run_evaluation(arguments.dataset, data_set, documents,
                                  feature_provider, activated_features,
                                  arguments.result_dir)

