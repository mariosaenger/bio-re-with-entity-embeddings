import pandas as pd

from pandas import DataFrame
from typing import Dict

from data.disease_ontology import DiseaseOntology
from data.resources import DO_ONTOLOGY_FILE, PHARMA_DIR, PHARMA_POS_PAIRS_FILE, PHARMA_NEG_PAIRS_FILE, \
    PHARMA_RELATIONSHIP_FILE
from utils.log_utils import LoggingMixin


class PharmaGKB(LoggingMixin):

    def __init__(self):
        super(PharmaGKB, self).__init__()

    def prepare(self, relationship_file: str, output_dir: str):
        # Load disease ontology for disease normalization
        disease_ontology = DiseaseOntology(DO_ONTOLOGY_FILE)

        md_relations = self.read_relationships(relationship_file)

        md_relations_pos = md_relations[md_relations["Association"] == "associated"]
        self.log_info(f"Found {len(md_relations_pos)} positive association entries")

        positive_pairs = self._prepair_pairs(md_relations_pos, disease_ontology)
        self.log_info(f"Found {len(positive_pairs)} valid positive pairs in total")

        md_relations_neg = md_relations[md_relations["Association"] == "not associated"]
        self.log_info(f"Found {len(md_relations_neg)} negative association entries")

        negative_pairs = self._prepair_pairs(md_relations_neg, disease_ontology)
        self.log_info(f"Found {len(negative_pairs)} valid negative pairs in total")

        self.log_info(f"Saving positive pairs to {PHARMA_POS_PAIRS_FILE}")
        self._save_pairs(positive_pairs, PHARMA_POS_PAIRS_FILE)

        self.log_info(f"Saving positive pairs to {PHARMA_NEG_PAIRS_FILE}")
        self._save_pairs(negative_pairs, PHARMA_NEG_PAIRS_FILE)

    def _prepair_pairs(self, relations: DataFrame, disease_ontology: DiseaseOntology) -> Dict:
        pairs = dict()
        unknown_diseases = set()

        for id, row in relations.iterrows():
            rs_id = row["Entity1_name"].lower()
            if not rs_id.startswith("rs"):
                continue

            disease_name = row["Entity2_name"].lower()
            doid = disease_ontology.get_doid_by_name(disease_name)
            if doid is None:
                unknown_diseases.add(disease_name)
                continue

            articles = row["PMIDs"]
            if articles is not None and type(articles) == str:
                articles = ",".join(articles.split(";"))
            else:
                articles = None

            pair = (rs_id, doid)
            if not pair in pairs:
                pairs[pair] = {
                    "source_id": rs_id,
                    "target_id": doid,
                    "disease": disease_name,
                    "articles":  articles,
                    "origin_id": id
                }
            else:
                self.log_warn(f"Found duplicate pair {pair}")

        return pairs

    def _save_pairs(self, pairs: Dict, output_file: str):
        with open(output_file, "w", encoding="utf-8") as writer:
            header = "\t".join(["source_id", "target_id", "disease", "articles", "origin_id"])
            writer.write(f"{header}\n")

            for (source_id, target_id), values in pairs.items():
                articles = values["articles"]
                if articles is None:
                    articles = ""

                writer.write(f"{source_id}\t{target_id}\t{values['disease']}\t{articles}\t{values['origin_id']}\n")

            writer.close()

    def read_relationships(self, input_file: str) -> DataFrame:
        self.log_info(f"Reading relationship from {input_file}")
        relations = pd.read_csv(input_file, sep="\t")
        self.log_info(f"Found {len(relations)} in total")

        md_relations = relations[(relations["Entity1_type"] == "Variant") & (relations["Entity2_type"] == "Disease")]
        self.log_info(f"Found {len(md_relations)} mutation-disease relations")

        return md_relations

    def read_positive_relationships(self, input_file: str) -> DataFrame:
        md_relations = self.read_relationships(input_file)

        md_relations_pos = md_relations[md_relations["Association"] == "associated"]
        self.log_info(f"Found {len(md_relations_pos)} positive association entries")

        return md_relations_pos


if __name__ == "__main__":
    pharma_gkb = PharmaGKB()
    pharma_gkb.prepare(PHARMA_RELATIONSHIP_FILE, PHARMA_DIR)
