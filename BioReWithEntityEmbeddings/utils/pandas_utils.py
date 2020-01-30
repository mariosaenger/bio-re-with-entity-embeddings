import numpy as np
import pandas as pd

from gensim.models import Word2Vec, Doc2Vec
from gensim.models.keyedvectors import Word2VecKeyedVectors
from nltk import word_tokenize

from pandas import DataFrame
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from typing import Callable, List, Dict, Any

from utils.log_utils import LoggingMixin


class PandasUtil(object):

    @staticmethod
    def count_values(column: str, target_column: str, splitter: str=None):
        return ValueCounter(column, target_column, splitter)

    @staticmethod
    def drop_duplicates(columns: List[str]):
        return DropDuplicates(columns)

    @staticmethod
    def extract_document_texts(document_column: str, text_column: str):
        return DocumentTextExtractor(document_column, text_column)

    @staticmethod
    def extract_unique_values(column:str, output_file:str, values_extractor=lambda x: x):
        return UniqueValueExtractor(column, output_file, values_extractor)

    @staticmethod
    def lookup_doc2vec_embeddings(doc2vec: Doc2Vec, key_extractor: Callable, column: str):
        return Doc2VecLookup(doc2vec, key_extractor, column)

    @staticmethod
    def map(src_column: str, map_function, tgt_column: str):
        return MapFunction(src_column, map_function, tgt_column)

    @staticmethod
    def not_null(column: str):
        return NotNull(column)

    @staticmethod
    def rename_columns(rename_map: dict):
        return RenameColumns(rename_map)

    @staticmethod
    def retrieve_documents(document_dictionary: Dict[str, Any], ids_column: str, documents_column, splitter: str =","):
        return DocumentRetriever(document_dictionary, ids_column, documents_column, splitter)

    @staticmethod
    def select_columns_by_prefixes(prefixes: List[str]):
        return ColumnPrefixDataFrameSelector(prefixes)

    @staticmethod
    def to_array():
        return DataFrameToArrayConverter()

    @staticmethod
    def to_avg_vector(word2vec: Word2VecKeyedVectors, text_column: str, vector_column: str):
        return TextToAvgVector(text_column, vector_column, word2vec)

    @staticmethod
    def to_bow(text_column: str, bow_column: str, vectorizer: CountVectorizer):
        return TextToBow(text_column, bow_column, vectorizer)

    @staticmethod
    def to_csv(output_file: str, separator="\t", columns: List[str]=None):
        return CsvWriter(output_file, separator=separator, columns=columns)


class PipelineMixin(TransformerMixin, LoggingMixin):

    def __init__(self):
        super(PipelineMixin, self).__init__()

    def fit(self, X, y=None):
        return self


class NullOperation(PipelineMixin):

    def __init__(self):
        super(NullOperation, self).__init__()

    def transform(self, data, Y=None):
        # Do nothing
        return data


class NotNull(PipelineMixin):

    def __init__(self, column: str, check_empty: bool = True):
        super(NotNull, self).__init__()
        self.column = column
        self.check_empty = check_empty

    def transform(self, data: DataFrame, y=None):
        self.log_info("Removing all entries with null value in column %s", self.column)
        original_size = len(data)

        data = data[data[self.column].notnull()]
        if self.check_empty:
            data = data[data[self.column].notna()]

        self.log_info("Removed %s entries (new size: %s)", original_size - len(data), len(data))
        return data


class MapFunction(PipelineMixin):

    def __init__(self, source_column: str, map_function, target_column: str):
        super(MapFunction, self).__init__()

        self.source_column = source_column
        self.map_function = map_function
        self.target_column = target_column

    def transform(self, data: DataFrame, y=None):
        data[self.target_column] = data[self.source_column].apply(self.map_function)
        return data


class ValueCounter(PipelineMixin):

    def __init__(self, column: str, target_column: str, splitter: str=","):
        super(ValueCounter, self).__init__()
        self.column = column
        self.target_column = target_column
        self.splitter = splitter

    def transform(self, data: DataFrame, y=None) -> DataFrame:
        data[self.target_column] = data[self.column].apply(self._count_values(self.splitter))
        return data

    @staticmethod
    def _count_values(splitter) -> Callable:
        def __count_values(value):
            if isinstance(value, list) or isinstance(value, set):
                return len(value)
            return len(str(value).split(splitter))
        return __count_values


class RenameColumns(PipelineMixin):

    def __init__(self, rename_map: dict):
        super(RenameColumns, self).__init__()
        self.rename_map = rename_map

    def transform(self, data: DataFrame):
        updated_col_names = [self.rename_map[column] if column in self.rename_map else column
                             for column in data.columns]
        data.columns = updated_col_names
        return data

class UniqueValueExtractor(PipelineMixin):

    def __init__(self, column: str, output_file: str, values_extractor: Callable):
        super(UniqueValueExtractor, self).__init__()

        self.column = column
        self.output_file = output_file
        self.values_extractor = values_extractor

    def transform(self, data, y=None):
        value_lists = data[self.column].map(self.values_extractor).values
        unique_values = set([value for value_list in value_lists for value in value_list
                             if value_lists is not None and value_list is not None])
        unique_values = list(set(unique_values))
        unique_values.sort()

        with open(self.output_file, 'w', encoding="utf-8") as output_file:
            output_file.writelines(["%s\n" % value for value in unique_values])
            output_file.close()

        return data


class CsvWriter(PipelineMixin):

    def __init__(self, output_file: str, separator: str = "\t", columns: List[str] = None):
        super(CsvWriter, self).__init__()

        self.output_file = output_file
        self.separator = separator
        self.columns = columns

    def transform(self, X, y = None):
        output_data_set = X if self.columns is None else X[self.columns]
        output_data_set.to_csv(self.output_file, sep=self.separator, index=False, encoding="utf-8")
        return X


class DropDuplicates(PipelineMixin):

    def __init__(self, columns: List[str]):
        super(DropDuplicates, self).__init__()
        self.columns = columns

    def transform(self, data: DataFrame, y=None):
        self.log_info("Removing duplicates according to columns: %s", ",".join(self.columns))
        clean_data = data.drop_duplicates(subset=self.columns)
        self.log_info("Removed %s duplicates", len(data) - len(clean_data))

        return clean_data


class EntryFilter(PipelineMixin):

    def __init__(self, pair_df: DataFrame, check_both: bool):
        super(EntryFilter, self).__init__()
        self.pair_df = pair_df
        self.check_both = check_both

        self.source_ids_map = None
        self.target_ids_map = None

    def transform(self, data, Y=None):
        self.log_info("Building auxiliary data structures")
        self.__initialize_maps()

        self.log_info(f"Start filtering entries based on {len(self.source_ids_map)} unique source "
                      f"and {len(self.target_ids_map)} unique target ids "
                      f"(checking both: {self.check_both})")
        invalid_ids = set()

        for id, row in tqdm(data.iterrows(), total=len(data)):
            source_id = row["source_id"]
            target_id = row["target_id"]

            # Both, source and target entity, must also be in the positive set!
            if self.check_both and (source_id not in self.source_ids_map or target_id not in self.target_ids_map):
                invalid_ids.add(id)

            #  Either the source or the target entity must be in the positive set!
            elif source_id not in self.source_ids_map and target_id not in self.target_ids_map:
                invalid_ids.add(id)

        self.log_info(f"Removing {len(invalid_ids)} from data set with {len(data)} entries")
        data = data.drop(invalid_ids)

        self.log_info(f"Filter data set contains {len(data)} instances")
        return data

    def __initialize_maps(self):
        self.source_ids_map = {id : True for id in self.pair_df["source_id"].unique()}
        self.target_ids_map = {id : True for id in self.pair_df["target_id"].unique()}


class DocumentRetriever(PipelineMixin):

    def __init__(self, document_dictionary: Dict, ids_column: str, documents_column: str, splitter: str =","):
        super(DocumentRetriever, self).__init__()
        self.document_dictionary = document_dictionary
        self.ids_column = ids_column
        self.documents_column = documents_column
        self.splitter = splitter

    def transform(self, X: DataFrame, y = None):
        X[self.documents_column] = X[self.ids_column].apply(self.retrieve_documents())
        return X

    def retrieve_documents(self) -> Callable:
        def _retrieve(value):
            document_ids = str(value).split(self.splitter)
            documents = [self.document_dictionary[id] for id in document_ids if id in self.document_dictionary]
            return documents
        return _retrieve


class DocumentTextExtractor(PipelineMixin):

    def __init__(self, document_column: str, text_column: str):
        super(DocumentTextExtractor, self).__init__()
        self.document_column = document_column
        self.text_column = text_column

    def transform(self, data: DataFrame, y=None):
        data[self.text_column] = data[self.document_column].apply(self.extract_document_text())
        return data

    def extract_document_text(self):
        def _extract(value):
            if isinstance(value, list):
                documents = value
            elif value:
                documents = [value]
            else:
                documents = []

            text = ""
            if len(documents) > 0:
                text = " ".join([document.text for document in documents])

            return text.strip()

        return _extract


class TextToBow(PipelineMixin):

    def __init__(self, text_column: str, bow_column: str, vectorizer: CountVectorizer):
        super(TextToBow, self).__init__()
        self.text_column = text_column
        self.bow_column = bow_column
        self.vectorizer = vectorizer

    def fit(self, X: DataFrame, y=None):
        texts = X[self.text_column].values
        self.vectorizer.fit(texts)
        return self

    def transform(self, X: DataFrame, y=None):
        texts = X[self.text_column].values
        result = self.vectorizer.transform(texts).toarray()

        number_of_words = len(self.vectorizer.get_feature_names())
        feature_names = [self.bow_column + "-" + str(i) for i in range(number_of_words)]
        bow_df = pd.DataFrame(result, index=X.index, columns=feature_names)

        return pd.concat([X, bow_df], axis=1)


class TextToAvgVector(PipelineMixin):

    def __init__(self, text_feature, vector_feature, word2vec_model):
        super(TextToAvgVector, self).__init__()

        self.text_feature = text_feature
        self.vector_feature = vector_feature
        self.word2vec_model = word2vec_model

    def transform(self, X, y=None):
        texts = X[self.text_feature].values

        vectors = np.zeros((len(X), self.word2vec_model.vector_size))
        for i in tqdm(range(len(texts)), desc="build-vectors", total=len(texts)):
            words = word_tokenize(texts[i])

            vector = np.zeros((self.word2vec_model.vector_size, 1))
            for word in words:
                if word in self.word2vec_model.vocab:
                    word_vector = self.word2vec_model[word].reshape((self.word2vec_model.vector_size,1))
                    vector = np.add(vector, word_vector)

            vector = np.divide(vector, len(words))
            vector = vector / np.linalg.norm(vector, ord=2)
            vectors[i] = vector.reshape((-1, self.word2vec_model.vector_size))

        feature_names = [self.vector_feature + "-" + str(i) for i in range(self.word2vec_model.vector_size)]
        vectors_df = pd.DataFrame(vectors, index=X.index, columns=feature_names)

        return pd.concat([X, vectors_df], axis=1)


class Doc2VecLookup(PipelineMixin):

    def __init__(self, doc2vec: Doc2Vec, key_extractor: Callable, column: str):
        super(Doc2VecLookup, self).__init__()
        self.doc2vec = doc2vec
        self.key_extractor = key_extractor
        self.column = column

    def transform(self, data: DataFrame, y=None):
        doc_vectors = np.zeros((len(data), self.doc2vec.vector_size))
        row_num = 0

        for i, row in tqdm(data.iterrows(), desc="lookup-doc-vectors", total=len(data)):
            id = self.key_extractor(row)
            if id in self.doc2vec.docvecs:
                doc_vectors[row_num] = self.doc2vec.docvecs[id]

            row_num = row_num + 1

        feature_names = [self.column + "_" + str(i) for i in range(self.doc2vec.vector_size)]
        vectors_df = pd.DataFrame(doc_vectors, index=data.index, columns=feature_names)
        return pd.concat([data, vectors_df], axis=1)


class ColumnPrefixDataFrameSelector(PipelineMixin):

    def __init__(self, column_prefixes):
        super(ColumnPrefixDataFrameSelector, self).__init__()
        self.column_prefixes = column_prefixes

    def transform(self, X, y=None):
        columns = [column for column in X.columns for prefix in self.column_prefixes if column.startswith(prefix)]
        return X[columns]


class DataFrameToArrayConverter(PipelineMixin):

    def transform(self, X, y=None):
        return X.values
