from typing import Callable, List

from pandas import DataFrame
from sklearn.base import TransformerMixin
from tqdm import tqdm

from utils.log_utils import LoggingMixin


class PandasUtil(object):

    @staticmethod
    def count_values(column: str, target_column: str, splitter: str=None):
        return ValueCounter(column, target_column, splitter)

    @staticmethod
    def drop_duplicates(columns: List[str]):
        return DropDuplicates(columns)

    @staticmethod
    def not_null(column: str):
        return NotNull(column)

    @staticmethod
    def extract_unique_values(column:str, output_file:str, values_extractor=lambda x: x):
        return UniqueValueExtractor(column, output_file, values_extractor)

    @staticmethod
    def map(src_column: str, map_function, tgt_column: str):
        return MapFunction(src_column, map_function, tgt_column)

    @staticmethod
    def rename_columns(rename_map: dict):
        return RenameColumns(rename_map)

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
