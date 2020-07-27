import os
import pandas as pd

from typing import List, Optional, Set
from pandas import DataFrame
from tqdm import tqdm

from data.resources import DO_DOID_FILE, DO_DIR, DO_ONTOLOGY_FILE
from utils.log_utils import LoggingMixin


class DiseaseOntology(LoggingMixin):

    def __init__(self, ontology_tsv: str):
        super(DiseaseOntology, self).__init__(self.__class__.__name__)
        self.ontology = pd.read_csv(ontology_tsv, sep="\t", encoding="utf8", index_col="id")
        self.mesh_to_doid_map = None
        self.synonym_to_doid_map = None

    def get_meshs(self, doid: str) -> List[str]:
        mesh_terms = None

        # First search doid directly within the index
        if doid in self.ontology.index:
            mesh_terms = self.ontology.loc[doid]["mesh_terms"]

        else:
            # Otherwise check alternative ids
            entry = self.ontology[self.ontology["alternative_ids"].str.contains(doid + "#")==True]
            if len(entry) > 0:
                mesh_terms = entry["mesh_terms"][0]

        if mesh_terms is not None and type(mesh_terms) == str:
            mesh_terms = mesh_terms.split(";")
        else:
            mesh_terms = []

        return mesh_terms

    def get_all_doids(self):
        return self.ontology.index.values

    def get_name_by_doid(self, doid):
        return self.ontology.loc[doid]["name"]

    def get_doid_by_name(self, name) -> Optional[str]:
        name_low = name.lower()
        result = self.ontology[self.ontology["name"] == name_low]
        if len(result) == 0:
            if self.synonym_to_doid_map is None:
                self._initialize_synonym_to_doid_map()

            if name_low in self.synonym_to_doid_map:
                return self.synonym_to_doid_map[name_low]

            return None

        return result["doid"].values[0]

    def get_doid_by_mesh(self, mesh: str) -> Set[str]:
        if self.mesh_to_doid_map is None:
            self._initialize_mesh_to_doid_map()

        if mesh in self.mesh_to_doid_map:
            return self.mesh_to_doid_map[mesh]
        else:
            return set()

    def get_paths(self, doid: str) -> set:
        disease_row = self.ontology.loc[doid]
        if len(disease_row) == 0:
            return set()

        parent_paths = disease_row["parent_paths"]
        paths = set()

        if not parent_paths == "[]":
            for parent_path in parent_paths.split(";"):
                path = parent_path + ">" + doid + "#"
                paths.add(path)

        return paths

    def get_path_prefixes_by_doid(self, doid: str, only_true_prefixes: bool = False) -> set:
        disease_row = self.ontology.loc[doid]
        if len(disease_row) == 0:
            return set()

        parent_paths = disease_row["parent_paths"]
        prefixes = set()

        if not parent_paths == "[]":
            for parent_path in parent_paths.split(";"):
                if len(parent_path) > 0:
                    path_components = parent_path.split(">")

                    prefix = path_components[0]
                    prefixes.add(prefix)

                    for component in path_components[1:]:
                        prefix = prefix + ">"  + component
                        prefixes.add(prefix)

                if not only_true_prefixes:
                    prefix = parent_path + ">" + doid + "#"
                    prefixes.add(prefix)
        else:
            prefixes.add(doid + "#")

        return prefixes

    def _initialize_mesh_to_doid_map(self):
        diseases_with_mesh = self.ontology[self.ontology["mesh_terms"].notnull()]
        self.mesh_to_doid_map = dict()

        for i, row in diseases_with_mesh.iterrows():
            for mesh in row["mesh_terms"].split(";"):
                if mesh not in self.mesh_to_doid_map:
                    self.mesh_to_doid_map[mesh] = set()
                self.mesh_to_doid_map[mesh].add(i)

    def _initialize_synonym_to_doid_map(self):
        self.log_info("Initializing synonym to doid mapping")
        self.synonym_to_doid_map = dict()

        for doid, row in tqdm(self.ontology.iterrows(), total=len(self.ontology)):
            synonyms = row["synonyms"]
            if not type(synonyms) is str or len(synonyms) == 0:
                continue

            for synonym in synonyms.lower().split(";"):
                if not synonym in self.synonym_to_doid_map:
                    self.synonym_to_doid_map[synonym] = doid
                else:
                    #self.log_warn(f"Found duplicate synonym {synonym}")
                    pass


class DiseaseOntologyHandler(LoggingMixin):

    def __init__(self):
        super(DiseaseOntologyHandler, self).__init__(self.__class__.__name__)

    def prepare_ontology(self, obo_file: str, output_dir: str) -> DataFrame:
        ontology_df = self.parse_obo_file(obo_file)
        ontology_df = self.append_paths(ontology_df)

        ontology_file = os.path.join(output_dir, "ontology.tsv")
        ontology_df.to_csv(ontology_file, sep="\t", encoding="utf8", index_label="id")

        return ontology_df

    def parse_obo_file(self, obo_file: str) -> DataFrame:
        ontology = pd.DataFrame(columns=["doid", "name", "alternative_ids", "parent_ids", "mesh_terms"])

        self.log_info("Reading disease ontology data from %s", obo_file)
        with open(obo_file, "r", encoding="utf-8") as do_reader:
            file_content = [line.strip() for line in do_reader.readlines()]

            primary_doid = None
            name = None
            alt_ids = set()
            parent_ids = set()
            mesh_terms = set()
            synonynms = set()

            for line in tqdm(file_content, total=len(file_content)):
                line = line.strip()

                if line.startswith("id: "):
                    primary_doid = self.clean_id(line.replace("id: ", ""))

                elif line.startswith("name:"):
                    name = line.replace("name:", "").lower().strip()

                elif line.startswith("synonym:"):
                    synonym = line.replace("synonym: ", "").lower().strip()
                    if not "exact [" in synonym or not synonym.startswith("\""):
                        #print(synonym)
                        continue

                    if ";" in synonym:
                        #print(synonym)
                        pass

                    name_end_index = synonym.find("\" exact []")
                    synonym = synonym.replace("\" exact []", "")[1:name_end_index]
                    synonynms.add(synonym)

                elif line.startswith("alt_id: "):
                    alt_ids.add(self.clean_id(line.replace("alt_id: ", "")))

                elif line.startswith("xref: MESH:"):
                    mesh_terms.add(line.replace("xref: ", "").strip())

                elif line.startswith("is_a: DOID"):
                    id_reference = line.replace("is_a: ", "").strip()
                    id_reference = id_reference[:id_reference.rfind("!")-1]
                    parent_ids.add(self.clean_id(id_reference))

                elif line.startswith("[Term]"):
                    if primary_doid is not None:
                        alt_ids = ";".join([str(id) + "#" for id in alt_ids]) if len(alt_ids) > 0 else None
                        parent_ids = ";".join(parent_ids) if len(parent_ids) > 0 else None
                        mesh_terms = ";".join(mesh_terms) if len(mesh_terms) > 0 else None

                        ontology = ontology.append({
                            "doid": primary_doid,
                            "name" : name,
                            "alternative_ids": alt_ids,
                            "parent_ids": parent_ids,
                            "mesh_terms": mesh_terms,
                            "synonyms": ";".join(synonynms)
                        },
                        ignore_index=True)

                    primary_doid = None
                    alt_ids = set()
                    parent_ids = set()
                    mesh_terms = set()
                    synonynms = set()

            do_reader.close()

        ontology.index = ontology["doid"]

        return ontology

    def append_paths(self, ontology_data: DataFrame) -> DataFrame:
        parent_paths = []
        parent_cache = dict()

        self.log_info("Appending paths to data set")
        for i, row in tqdm(ontology_data.iterrows(), total=len(ontology_data)):
            parent_ids = row["parent_ids"]
            if parent_ids is None:
                parent_paths.append([])
                continue

            all_parent_paths = []
            for parent_id in str(row["parent_ids"]).split(";"):
                if parent_id in parent_cache:
                    paths = parent_cache[parent_id]
                else:
                    paths = self.get_parent_paths(ontology_data, parent_id)
                    parent_cache[parent_id] = paths

                all_parent_paths = all_parent_paths + paths

            if len(all_parent_paths) > 0:
                parent_paths.append(";".join([">".join(path) for path in all_parent_paths]))
            else:
                parent_paths.append(None)

        ontology_data["parent_paths"] = parent_paths
        return ontology_data

    def get_parent_paths(self, ontology: DataFrame, doid: str) -> List[List[str]]:
        entry = ontology.loc[ontology["doid"] == doid]
        if len(entry) == 0:
            return [[]]

        doid_with_end_marker = doid + "#"
        parent_ids = str(entry["parent_ids"].values[0]).split(";")
        if len(parent_ids) == 0:
            return [[doid_with_end_marker]]

        # A disease node can have multiple parents
        #   -> get the path for each parent and append the current disease id to it
        return [path + [doid_with_end_marker]
                for parent_id in parent_ids
                for path in self.get_parent_paths(ontology, parent_id)
                if parent_id is not None]

    def clean_id(self, id: str):
        id = id.replace("DOID:", "").strip()
        try:
            id = str(int(id))
        except:
            pass

        return "DOID:" + id


if __name__ == "__main__":
    handler = DiseaseOntologyHandler()
    ontology = handler.prepare_ontology(DO_DOID_FILE, DO_DIR)

