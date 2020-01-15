import numpy as np
import os
import pandas as pd
from docutils.nodes import Sequential

from gensim.models import KeyedVectors, Doc2Vec
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.initializers import VarianceScaling
from keras.layers import Dense, BatchNormalization, Dropout
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from typing import List, Dict

from data.pubtator import Document, Pubtator
from utils.log_utils import LoggingMixin
from utils.pandas_utils import PandasUtil as pdu


class EvaluationResult(object):

    def __init__(self, pred_probabilities: np.ndarray, pred_labels: np.ndarray, accuracy_score: float, f1_score: float):
        self.pred_probabilities = pred_probabilities
        self.pred_labels = pred_labels
        self.accuracy_score = accuracy_score
        self.f1_score = f1_score


class DataSetReader(LoggingMixin):

    def __init__(self, working_dir: str):
        super(DataSetReader, self).__init__(self.__class__.__name__)
        self.working_dir = working_dir

    def read_data_set(self) -> DataFrame:
        pos_sample_file = os.path.join(self.working_dir, "pos_samples.tsv")
        pos_samples = self._read_samples(pos_sample_file, 1)

        neg_sample_file = os.path.join(self.working_dir, "neg_samples.tsv")
        neg_samples = self._read_samples(neg_sample_file, 0)

        if pos_samples is not None and neg_samples is not None:
            data_set = pd.concat([pos_samples, neg_samples])
        elif pos_samples is not None:
            data_set = pos_samples
        elif neg_samples is not None:
            data_set = neg_samples
        else:
            raise AssertionError(f"Can't find any samples!")

        data_set = data_set.reset_index(drop=True)
        self.log_info("Complete data set contains %s instances", len(data_set))

        return data_set

    def _read_samples(self, input_file:str, label: int):
        if os.path.exists(input_file):
            self.log_info("Start loading data set from %s", input_file)
            samples = pd.read_csv(input_file, "\t", encoding="utf-8")

            if len(samples.columns) == 4:
                samples.columns = ["source_id", "target_id", "source", "articles"]
            else:
                samples.columns = ["source_id", "target_id", "articles"]

            samples["label"] = label
            self.log_info("Found %s instances", len(samples))
        else:
            samples = None
            self.log_warn(f"Can't find input file {input_file}")

        return samples

    def read_documents(self) -> Dict[str, Document]:
        pos_article_file = os.path.join(self.working_dir, "pos_articles.txt")
        neg_article_file = os.path.join(self.working_dir, "neg_articles.txt")

        pubtator = Pubtator()

        self.log_info("Start loading documents from %s", pos_article_file)
        pos_documents = pubtator.read_documents_from_file(pos_article_file)
        self.log_info("Found %s documents", len(pos_documents))

        self.logger.info("Start loading documents from %s", neg_article_file)
        neg_documents = pubtator.read_documents_from_file(neg_article_file)
        self.log_info("Found %s documents", len(neg_documents))

        self.log_info("Merge document sets")
        documents = list(set(pos_documents + neg_documents))
        self.log_info("Found %s documents in total", len(documents))

        return dict([(str(document.id), document) for document in documents])


class FeatureConfigurationProvider():

    def create_feature_configurations(self, activated_features: List):
        configurations = []

        if "entity" in activated_features:
            configurations = configurations + self._create_entity_configurations()

        if "pair" in activated_features:
            configurations = configurations + self._create_pair_configurations()

        return configurations

    def _create_entity_configurations(self) -> List:
        raise NotImplementedError()

    def _create_pair_configurations(self) -> List:
        raise NotImplementedError()


class MutationDiseaseFeatureConfigurationProvider(FeatureConfigurationProvider):

    def __init__(self, embedding_dir:str):
        super(MutationDiseaseFeatureConfigurationProvider, self).__init__()
        self.embedding_dir = embedding_dir

    def _create_entity_configurations(self) -> List:
        return [
            ("EntityDoc2Vec-v0500", "entity_doc2vec",{
                "source_entity_doc2vec_model": os.path.join(self.embedding_dir, "mutation-v0500.embs"),
                "target_entity_doc2vec_model": os.path.join(self.embedding_dir, "disease-v0500.embs")
            }),
            ("EntityDoc2Vec-v1000", "entity_doc2vec",{
                "source_entity_doc2vec_model": os.path.join(self.embedding_dir, "mutation-v1000.embs"),
                "target_entity_doc2vec_model": os.path.join(self.embedding_dir, "disease-v1000.embs")
            }),
            ("EntityDoc2Vec-v1500", "entity_doc2vec", {
                "source_entity_doc2vec_model": os.path.join(self.embedding_dir, "mutation-v1500.embs"),
                "target_entity_doc2vec_model": os.path.join(self.embedding_dir, "disease-v1500.embs")
            }),
            ("EntityDoc2Vec-v2000", "entity_doc2vec",{
                "source_entity_doc2vec_model": os.path.join(self.embedding_dir, "mutation-v2000.embs"),
                "target_entity_doc2vec_model": os.path.join(self.embedding_dir, "disease-v2000.embs")
            }),
        ]

    def _create_pair_configurations(self) -> List:
        return [
            ("PairDoc2Vec-v500", "pair_doc2vec", {
                "pair_doc2vec_model": os.path.join(self.embedding_dir, "mutation-disease-v0500.embs")
            }),
            ("PairDoc2Vec-v1000", "pair_doc2vec", {
                "pair_doc2vec_model": os.path.join(self.embedding_dir, "mutation-disease-v1000.embs")
            }),
            ("PairDoc2Vec-v1500", "pair_doc2vec", {
                "pair_doc2vec_model": os.path.join(self.embedding_dir, "mutation-disease-v1500.embs")
            }),
            ("PairDoc2Vec-v2000", "pair_doc2vec", {
                "pair_doc2vec_model": os.path.join(self.embedding_dir, "mutation-disease-v0500.embs")
            }),
        ]


class DrugDrugFeatureConfigurationProvider(FeatureConfigurationProvider):

    def __init__(self, embedding_dir:str):
        super(DrugDrugFeatureConfigurationProvider, self).__init__()
        self.embedding_dir = embedding_dir

    def _create_entity_configurations(self) -> List:
        entity_v0500_model = os.path.join(self.embedding_dir, "drug-v0500.embs")
        entity_v1000_model = os.path.join(self.embedding_dir, "drug-v1000.embs")
        entity_v1500_model = os.path.join(self.embedding_dir, "drug-v1500.embs")
        entity_v2000_model = os.path.join(self.embedding_dir, "drug-v2000.embs")

        return [
            ("EntityDoc2Vec-v0500", "entity_doc2vec",{
                    "source_entity_doc2vec_model": entity_v0500_model,
                    "target_entity_doc2vec_model": entity_v0500_model
            }),
            ("EntityDoc2Vec-v1000", "entity_doc2vec",{
                "source_entity_doc2vec_model": entity_v1000_model,
                "target_entity_doc2vec_model": entity_v1000_model
            }),
            ("EntityDoc2Vec-v1500", "entity_doc2vec",{
                "source_entity_doc2vec_model": entity_v1500_model,
                "target_entity_doc2vec_model": entity_v1500_model
            }),
            ("EntityDoc2Vec-v2000", "entity_doc2vec",{
                "source_entity_doc2vec_model": entity_v2000_model,
                "target_entity_doc2vec_model": entity_v2000_model
            }),
        ]

    def _create_pair_configurations(self) -> List:
        return [
            ("PairDoc2Vec-v0500", "pair_doc2vec", {
                "pair_doc2vec_model": os.path.join(self.embedding_dir, "drug-drug-v0500.embs")
            }),
            ("PairDoc2Vec-v1000", "pair_doc2vec", {
                "pair_doc2vec_model": os.path.join(self.embedding_dir, "drug-drug-v1000.embs")
            }),
            ("PairDoc2Vec-v1500", "pair_doc2vec", {
                "pair_doc2vec_model": os.path.join(self.embedding_dir, "drug-drug-v1500.embs")
            }),
            ("PairDoc2Vec-v2000", "pair_doc2vec", {
                "pair_doc2vec_model": os.path.join(self.embedding_dir, "drug-drug-v2000.embs")
            })
        ]


class DatePreparationPipeline(LoggingMixin):

    def __init__(self):
        super(DatePreparationPipeline, self).__init__()

    def get_data_pipeline(self, feature: str, configuration: Dict, document_dict: Dict[str, Document]):
        feature_columns = []
        pipeline_steps = []

        if feature in ("bow", "tfidf", "pubmed_vector"):
            pipeline_steps.append(("RetrieveDocuments",
                                   pdu.retrieve_documents(document_dict, "articles", "documents", ";;;")))
            pipeline_steps.append(("ExtractDocumentTexts",
                                   pdu.extract_document_texts("documents", "text")))

        if feature == "bow":
            if "bow_vectorizer" in configuration:
                vectorizer = configuration["bow_vectorizer"]
            else:
                vectorizer = CountVectorizer(binary=True, dtype=np.float, **configuration)
                configuration["bow_vectorizer"] = vectorizer

            pipeline_steps.append(("TextToBow", pdu.to_bow("text", "bow", vectorizer)))
            feature_columns.append("bow")

        if feature == "tfidf":
            if "tfidf_vectorizer" in configuration:
                vectorizer = configuration["tfidf_vectorizer"]
            else:
                vectorizer = CountVectorizer(ngram_range=(1, 1), dtype=np.float32, **configuration)
                configuration["tfidf_vectorizer"] = vectorizer

            pipeline_steps.append(("TextToTfIdf", pdu.to_bow("text", "tfidf", vectorizer)))
            feature_columns.append("tfidf")

        elif feature == "pubmed_vector":
            self.log_info("Loading word2vec model")
            pubmed_model = Embeddings().get_truncated_pub_med_model("PubMed-and-PMC-w2v_truncated.vec")

            column_name = "pubmed_vector_" + configuration["operation"]
            pipeline_steps.append(("TextToAvgVector", pdu.to_avg_vector(pubmed_model, "text", column_name)))
            feature_columns.append(column_name)

        elif feature == "pair_doc2vec":
            embedding_file = configuration["pair_doc2vec_model"]
            doc2vec_pair_embeddings = Embeddings().load(embedding_file)

            key_extractor = lambda row: str(row["source_id"]) + ";;;" + str(row["target_id"])

            pipeline_steps.append(("LookupEntityPairEmbeddings", pdu.lookup_doc2vec_embeddings(
                  doc2vec_pair_embeddings, key_extractor, "entity_pair_embedding")))
            feature_columns.append("entity_pair_embedding")

        elif feature == "entity_doc2vec":
            source_entity_embedding_file = configuration["source_entity_doc2vec_model"]
            doc2vec_source = Embeddings().load(source_entity_embedding_file)

            target_entity_embedding_file = configuration["target_entity_doc2vec_model"]
            doc2vec_target = Embeddings().load(target_entity_embedding_file)

            source_key_extractor = lambda row: str(row["source_id"])
            source_embedding_lookup = pdu.lookup_doc2vec_embeddings(doc2vec_source, source_key_extractor, "source_embedding")
            pipeline_steps.append(("SourceEmbeddingLookup", source_embedding_lookup))

            target_key_extractor = lambda row: str(row["target_id"])
            target_embedding_lookup = pdu.lookup_doc2vec_embeddings(doc2vec_target, target_key_extractor, "target_embedding")
            pipeline_steps.append(("TargetEmbeddingLookup", target_embedding_lookup))

            feature_columns = feature_columns + ["source_embedding", "target_embedding"]

        pipeline_steps.append(("FeatureSelection", pdu.select_columns_by_prefixes(feature_columns)))
        pipeline_steps.append(("DataFrameConverter", pdu.to_array()))

        return Pipeline(pipeline_steps)

    def get_label_pipeline(self, label="label"):
        return Pipeline([
            ("feature_selector", pdu.select_columns_by_prefixes(label)),
            ("data_frame_converter", pdu.to_array())
        ])


class EvaluationUtils():

    @staticmethod
    def save_predictions(data_set: DataFrame, pred_probabilities, pred_labels, output_file: str) -> None:
        instances_pd = data_set[["source_id", "target_id", "label"]]
        probabilities_pd = pd.DataFrame(pred_probabilities, columns=["prob0", "prob1"], index=instances_pd.index)
        prediction_pd = pd.DataFrame(pred_labels, columns=["prediction"], index=instances_pd.index)

        complete_pd = pd.concat([instances_pd, prediction_pd, probabilities_pd], axis=1)
        complete_pd.to_csv(output_file, sep="\t", index=False)


class Embeddings(LoggingMixin):

    def __init__(self):
        super(Embeddings, self).__init__()

    def get_truncated_pub_med_model(self, path: str):
        self.log_info(f"Start loading word2vec model from {path}")
        return KeyedVectors.load_word2vec_format(path)

    def load(self, path: str):
        self.log_info("Start loading doc2vec model from %s", path)
        doc2vec_model = Doc2Vec.load(path)

        self.log_info("Finished loading. Model contains %s documents vectors and a vocabulary of %s terms",
                      len(doc2vec_model.docvecs), len(doc2vec_model.wv.vectors))

        return doc2vec_model


class MultiLayerPerceptron(LoggingMixin):

    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()

    def create_model(self, input_features: int, target_classes: int, hidden_layer_sizes: List[int],
                     drop_out: float=None, batch_normalization: bool = False, validation_data = None,
                     epochs: int = 500, batch_size: int = 16, callbacks=None):

        def _build_model():
            model = Sequential()

            for i, layer_size in enumerate(hidden_layer_sizes):
                if i == 0:
                    model.add(Dense(layer_size, input_shape=(input_features,),
                                    kernel_initializer=VarianceScaling(seed=773),
                                    bias_initializer="ones",
                                    activation="selu"))
                else:
                    model.add(Dense(layer_size,
                                    kernel_initializer=VarianceScaling(seed=773),
                                    bias_initializer="ones",
                                    activation="selu"))

                if batch_normalization:
                    model.add(BatchNormalization())

                if drop_out is not None:
                        model.add(Dropout(drop_out, seed=773))


            model.add(Dense(target_classes, activation="softmax",
                            kernel_initializer=VarianceScaling(seed=773),
                            bias_initializer="ones"))
            model.compile(optimizer=Adam(lr=0.0001, decay=0.0008),
                          loss="sparse_categorical_crossentropy",
                          metrics=['accuracy'])
            model.summary(print_fn=self.log_info)

            return model

        return KerasClassifier(_build_model, epochs=epochs, batch_size=batch_size,
                               validation_data=validation_data, callbacks=callbacks)

    @staticmethod
    def create_mlp_model(X_val: np.ndarray, y_val: np.ndarray, layer_config: List[int], do: float = None,
                         bn: bool = False, epochs: int = 250, batch_size: int = 4):
        dnn_callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, verbose=1),
            ModelCheckpoint("mlp_model.tmp", save_best_only=True, verbose=1)
        ]

        K.clear_session()
        gc.collect()

        mlp_classifier = MultiLayerPerceptron()
        return mlp_classifier.create_model(X_val.shape[1], 2, layer_config, do, bn,
                                           (X_val, y_val), epochs, batch_size, dnn_callbacks)
