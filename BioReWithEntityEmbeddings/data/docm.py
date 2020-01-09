import pandas as pd

from tqdm import tqdm
from typing import Dict

from data.disease_ontology import DiseaseOntology
from data.resources import DOCM_VARIANTS, DOCM_HGVS_MAPPING, RSID_TO_HGVS_FILE, DOCM_DATA_SET_FILE, DO_ONTOLOGY_FILE
from utils.log_utils import LoggingMixin
from utils.mapping_utils import RsidToHgvsMapper


class DoCM(LoggingMixin):

    def __init__(self):
        super(DoCM, self).__init__(self.__class__.__name__)

    def prepare(self, docm_variants_file: str, hgvs_mapping_file: str, rsid_mapping_file: str, output_file: str):
        self.log_info(f"Reading variants from {docm_variants_file}")
        docm_df = pd.read_csv(docm_variants_file, sep="\t", encoding="utf8")

        # Read hgvs to normalized hgvs mapping (e.g. ENST000123 -> NM_123456)
        hgvs_mapping = self.read_hgvs_mapping(hgvs_mapping_file)

        # Read mapping from hgvs to rsids (e.g. NM_123456 -> rs123)
        rsid_mapper = RsidToHgvsMapper()
        hgvs_to_rsid = rsid_mapper.read_hgvs_to_rsid_mapping(rsid_mapping_file)

        # Read disease ontology for mappings of disease name to doid (e.g. breast cancer -> DOID:1234)
        disease_ontology = DiseaseOntology(DO_ONTOLOGY_FILE)

        instances = dict()
        for row_id, row in tqdm(docm_df.iterrows(), total=len(docm_df)):
            hgvs = row["hgvs"]

            if hgvs not in hgvs_mapping:
                self.log_warn(f"Can't find normalized version of {hgvs}")
                continue

            normalized_hgvs = hgvs_mapping[hgvs]
            if normalized_hgvs not in hgvs_to_rsid:
                self.log_warn(f"Can't find rs id for {normalized_hgvs} ({hgvs})")
                continue

            rs_id = hgvs_to_rsid[normalized_hgvs]

            diseases = row["diseases"].lower().split(",")
            for disease in diseases:
                doid = disease_ontology.get_doid_by_name(disease)
                if doid is None:
                    self.log_warn(f"Can't find doid for {disease}")
                    continue

                articles = set(row["pubmed_sources"].split(","))
                if (rs_id, doid) not in instances:
                    pair_information = {
                        "source_id": rs_id,
                        "gene": row["gene"],
                        "hgvs": hgvs,
                        "target_id": doid,
                        "disease": disease,
                        "articles": articles,
                        "origin_id": row_id
                    }
                    instances[(rs_id, doid)] = pair_information
                else:
                    # Add articles to existing entry
                    instances[(rs_id, doid)]["articles"].update(articles)

        self.log_info(f"Saving instances to {output_file}")
        with open(output_file, "w", encoding="utf8") as writer:
            writer.write("\t".join(["source_id", "gene", "hgvs", "target_id", "disease", "origin_id", "articles"]) + "\n")

            for key, value in instances.items():
                writer.write("\t".join([
                    value["source_id"],
                    value["gene"],
                    value["hgvs"],
                    value["target_id"],
                    value["disease"],
                    str(value["origin_id"]),
                    ",".join(value["articles"])
                ]))
                writer.write("\n")

        self.log_info(f"Finished preparation. Found {len(instances)} in total")

    def read_hgvs_mapping(self, input_file: str) -> Dict[str, str]:
        self.log_info(f"Reading hgvs mapping from {input_file}")
        hgvs_mapping = dict()
        with open(input_file, "r", encoding="utf8") as reader:
            for line in reader.readlines():
                hgvs, normalized_hgvs = line.strip().split("\t")
                hgvs_mapping[hgvs] = normalized_hgvs
            reader.close()

        self.log_info(f"Found {len(hgvs_mapping)} hgvs mappings")
        return hgvs_mapping


if __name__ == "__main__":
    docm = DoCM()
    docm.prepare(DOCM_VARIANTS, DOCM_HGVS_MAPPING, RSID_TO_HGVS_FILE, DOCM_DATA_SET_FILE)
