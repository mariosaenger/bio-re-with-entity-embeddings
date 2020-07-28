import pandas as pd

from pandas import DataFrame
from sklearn.base import TransformerMixin
from tqdm import tqdm
from typing import Callable, List

from data.disease_ontology import DiseaseOntology
from utils.log_utils import LoggingMixin


class PandasUtil(object):

    @staticmethod
    def extract_unique_values(column:str, output_file: str, values_extractor=lambda x: x):
        return UniqueValueExtractor(column, output_file, values_extractor)

    @staticmethod
    def map(src_column: str, map_function, tgt_column: str):
        return MapFunction(src_column, map_function, tgt_column)

    @staticmethod
    def not_null(column: str):
        return NotNull(column)

    @staticmethod
    def rename_columns(rename_map: dict):
        return RenameColumns(rename_map)


class PipelineMixin(TransformerMixin, LoggingMixin):

    def __init__(self):
        super(PipelineMixin, self).__init__()

    def fit(self, X, y=None):
        return self


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


class DropDuplicates(PipelineMixin):

    def __init__(self, columns: List[str]):
        super(DropDuplicates, self).__init__()
        self.columns = columns

    def transform(self, data: DataFrame, y=None):
        self.log_info("Removing duplicates according to columns: %s", ",".join(self.columns))
        clean_data = data.drop_duplicates(subset=self.columns)
        self.log_info("Removed %s duplicates", len(data) - len(clean_data))

        return clean_data


class MeshTermToDoidMapper(PipelineMixin):

    def __init__(self, disease_ontology: DiseaseOntology):
        super(MeshTermToDoidMapper, self).__init__()
        self.disease_ontology = disease_ontology

    def transform(self, data: DataFrame, y=None):
        self.log_info(f"Adding DOID id to {len(data)} instances")
        num_unknown_doid = 0

        new_data_map = dict()
        for mesh, row in tqdm(data.iterrows(), total=len(data)):
            doids = self.disease_ontology.get_doid_by_mesh(mesh)
            if len(doids) > 0:
                for doid in doids:
                    row_copy = row.copy()
                    row_copy["doid"] = doid

                    if doid in new_data_map:
                        a1 = row_copy["articles"]
                        a2 = new_data_map[doid]["articles"]
                        row_copy["articles"] = a1.union(a2)

                    new_data_map[doid] = row_copy
            else:
                num_unknown_doid = num_unknown_doid + 1

        new_data = DataFrame(list(new_data_map.values()))
        new_data.index = new_data["doid"]

        #self.log_info("Can't find DOID for %s of %s entries", num_unknown_doid, len(data))
        self.log_info(f"Finished MeSH to DOID mapping. "
                      f"New data set has {len(new_data)} instances")

        return new_data


class MeshTermToDrugBankIdMapper(PipelineMixin):

    def __init__(self, mesh_to_drugbank: pd.DataFrame):
        super(MeshTermToDrugBankIdMapper, self).__init__()
        self.mesh_to_drugbank = mesh_to_drugbank

    def transform(self, data: pd.DataFrame, y=None):
        self.log_info(f"Adding DrugBank identifier to {len(data)} instances")

        new_data_map = dict()
        for mesh, row in tqdm(data.iterrows(), total=len(data)):
            drugbank_ids = self.get_drugbank_ids_by_mesh(mesh)

            if len(drugbank_ids) > 0:
                for db_id in drugbank_ids:
                    row_copy = row.copy()
                    row_copy["drugbank_id"] = db_id

                    if db_id in new_data_map:
                        a1 = row_copy["articles"]
                        a2 = new_data_map[db_id]["articles"]
                        row_copy["articles"] = a1.union(a2)

                    new_data_map[db_id] = row_copy

        new_data = pd.DataFrame(list(new_data_map.values()))
        new_data.index = new_data["drugbank_id"]

        self.log_info(f"Finished MeSH to DrugBank id mapping. "
                      f"New data set has {len(data)} instances")

        return new_data

    def get_drugbank_ids_by_mesh(self, mesh: str) -> List[str]:
        if mesh in self.mesh_to_drugbank.index:
            return self.mesh_to_drugbank.loc[mesh]["DrugBankIDs"].split("|")

        return []
