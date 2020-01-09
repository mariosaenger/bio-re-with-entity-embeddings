import pandas as pd

from pandas import DataFrame
from tqdm import tqdm

from utils.log_utils import LoggingMixin


class Pubtator(LoggingMixin):

    def __init__(self):
        super(Pubtator, self).__init__()

    def build_disease_overview(self, disease2pubtator_file: str)-> DataFrame:
        disease_data = pd.read_csv(disease2pubtator_file, sep="\t", encoding="utf-8", memory_map=True)
        disease_overview = self._build_overview(disease_data, "MeshID")
        disease_overview["mesh_term"] = disease_overview.index

        return disease_overview

    def build_mutation_overview(self, mutation2pubtator_file: str) -> DataFrame:
        rs_mutation_data = self.read_mutation2pubtator_file(mutation2pubtator_file)

        mutation_overview = self._build_overview(rs_mutation_data, "Components")
        mutation_overview["rs_identifier"] = mutation_overview.index

        return mutation_overview

    def read_mutation2pubtator_file(self, mutation2pubtator_file: str) -> DataFrame:
        mutation_data = pd.read_csv(mutation2pubtator_file, sep="\t", encoding="utf-8", memory_map=True)

        non_tmVar_data = mutation_data.loc[mutation_data["Resource"] != "tmVar"]
        self.log_info("Found %s mutation mentions not tagged by tmVar", len(non_tmVar_data))

        tmVar_data = mutation_data.loc[mutation_data["Resource"] == "tmVar"]
        self.log_info("Found %s mutation mentions tagged by tmVar", len(tmVar_data))

        tmVar_rs_data = tmVar_data.loc[tmVar_data["Components"].str.contains("RS#:")]
        tmVar_rs_data["Components"] = tmVar_rs_data["Components"].map(Pubtator._clean_tmVar_components)
        self.log_info("Found %s mutation mentions with rs-identifier and tagged by tmVar", len(non_tmVar_data))

        rs_mutation_data = pd.concat([non_tmVar_data, tmVar_rs_data])
        rs_mutation_data = rs_mutation_data.loc[rs_mutation_data["Components"].str.startswith("rs")]
        self.log_info("Found %s mutation instances with rs-identifier", len(rs_mutation_data))

        return rs_mutation_data

    def _build_overview(self, data_set: DataFrame, id_column: str) -> DataFrame:
        ids = []
        articles = []

        group_by_id = data_set.groupby([id_column])
        for id, rows in tqdm(group_by_id, total=len(group_by_id)):
            ids.append(id)
            articles.append(set(rows["PMID"].unique()))

        return pd.DataFrame({"articles": articles}, index=ids)

    @staticmethod
    def _clean_tmVar_components(value: str) -> str:
        components = value.split(";")
        if len(components) == 2:
            rs_identifier = components[1].replace("RS#:", "rs")
            index = rs_identifier.rfind("|")
            if index != -1:
                rs_identifier = rs_identifier[:index]

            return rs_identifier.strip()

        return value
