import re

from concurrent.futures.thread import ThreadPoolExecutor
from typing import List, Dict
import pandas as pd

from pandas import DataFrame
from tqdm import tqdm

from utils.log_utils import LoggingMixin


class Document(object):

    def __init__(self, id: str, title: str, abstract: str):
        self.id = id
        self.title = title
        self.abstract = abstract

    def text(self):
        complete_text = self.title if self.title else ""
        complete_text = complete_text + " " + self.abstract if self.abstract else complete_text
        complete_text = complete_text.replace("\t", "")

        return complete_text.strip()


class Pubtator(LoggingMixin):

    TITLE_PATTERN = re.compile("([0-9]+)\\|t\\|(.*)")
    ABSTRACT_PATTERN = re.compile("([0-9]+)\\|a\\|(.*)")

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

    def read_raw_documents_from_offsets(self, offsets_file: str) -> List[str]:
        documents = list()
        with open(offsets_file, "r", encoding="utf-8") as input_reader:

            document = None
            all_lines = input_reader.readlines()
            for line in tqdm(all_lines, desc="read-documents", total=len(all_lines)):
                line = line.strip()

                if not line:
                    if document:
                        documents.append(document)
                        document = None

                if not document:
                    document = ""

                document = document + "\n" + line
            input_reader.close()

        return documents

    def parse_raw_documents_parallel(self, raw_documents: List[str], threads: int, batch_size: int) -> Dict[str, Document]:
        self.log_info("Start parsing % documents with %s threads and %s documents / thread", len(raw_documents), threads, batch_size)

        parsed_documents = dict()
        with ThreadPoolExecutor(max_workers=threads) as executor:
            self.log_info("Creating parsing %s jobs", int(len(raw_documents) / batch_size))
            futures = []

            for i in tqdm(range(0, len(raw_documents), batch_size), desc="create-jobs", total=len(raw_documents) / batch_size):
                document_batch = raw_documents[i:i + batch_size]
                future = executor.submit(Pubtator.parse_raw_documents, document_batch)
                futures.append(future)

            self.log_info("Collecting parse results")
            for future in tqdm(futures, desc="collect-result", total=len(futures)):
                parsed_documents.update(future.result())

            self.log_info("Finished document parsing")
            executor.shutdown()

        return parsed_documents

    @staticmethod
    def parse_raw_documents(raw_documents: List[str]) -> Dict[str, Document]:
        documents = dict()

        for raw_document in raw_documents:
            pubmed_id = None
            title = None
            abstract = None

            for line in raw_document.split("\n"):
                line = line.strip()
                if not line:
                    continue

                title_match = Pubtator.TITLE_PATTERN.match(line)
                if title_match:
                    pubmed_id = title_match.group(1)
                    title = title_match.group(2)
                    continue

                abstract_match = Pubtator.ABSTRACT_PATTERN.match(line)
                if abstract_match:
                    pubmed_id = abstract_match.group(1)
                    abstract = abstract_match.group(2)
                    continue

            documents[pubmed_id] = Document(pubmed_id, title, abstract)

        return documents

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
