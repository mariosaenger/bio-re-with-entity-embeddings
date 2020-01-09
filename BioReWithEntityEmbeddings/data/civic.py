import pandas as pd

from data.resources import CIVIC_EVIDENCES_FILE, CIVIC_VARIANT_SUMMARY_FILE, RSID_TO_HGVS_FILE, CIVIC_DATA_SET_FILE
from utils.log_utils import LoggingMixin
from utils.mapping_utils import RsidToHgvsMapper


class CIViC(LoggingMixin):

    def __init__(self):
        super(CIViC, self).__init__(self.__class__.__name__)

    def read_evidences(self, evidence_file: str, variant_file: str):
        self.log_info("Reading civic evidences from %s", evidence_file)
        evidences = pd.read_csv(evidence_file, sep="\t", index_col="evidence_id")
        self.log_info("Found %s evidences", len(evidences))

        self.log_info("Reading civic variant summaries from %s", variant_file)
        variant_summaries = pd.read_csv(variant_file, sep="\t")
        self.log_info("Found %s variants", len(variant_summaries))

        self.log_info("Joining evidences and variant summaries")
        evidences = evidences.merge(variant_summaries, on="variant_id")
        self.log_info("Found %s evidences after join", len(evidences))

        evidences = evidences.loc[evidences["hgvs_expressions"].notnull()]
        self.log_info(f"Found {len(evidences)} with hgvs expressions")

        evidences = evidences.loc[evidences["doid"].notnull()]
        self.log_info(f"Found {len(evidences)} with doid")

        return evidences

    def prepare(self, evidence_file: str, variant_file: str, rsid_mapping_file: str, output_file: str):
        # Read mapping from hgvs to rsids (e.g. NM_123456 -> rs123)
        rsid_mapper = RsidToHgvsMapper()
        hgvs_to_rsid = rsid_mapper.read_hgvs_to_rsid_mapping(rsid_mapping_file)

        evidences = self.read_evidences(evidence_file, variant_file)

        mutation_disease_dict = dict()
        for row_id, row in evidences.iterrows():
            disease_id = str(int(row["doid"]))

            rs_id = None
            for hgvs in row["hgvs_expressions"].split(","):
                transcript, change = hgvs.split(":")
                transcript, version = transcript.split(".")[0] if "." in transcript else transcript, ""

                normalized_hgvs = transcript + ":" + change
                if normalized_hgvs in hgvs_to_rsid:
                    rs_id = hgvs_to_rsid[normalized_hgvs]
                    break

            if rs_id is None:
                self.log_warn(f"Can't find rs-id for hgvs expressions: {row['hgvs_expressions']}")
                continue

            if (rs_id, disease_id) not in mutation_disease_dict:
                mutation_disease_dict[(rs_id, disease_id)] = {
                    "source_id" : rs_id,
                    "transcript" : row["representative_transcript_x"],
                    "gene" : row["gene_x"],
                    "target_id" : disease_id,
                    "articles" : set(),
                    "hgvs_expressions" : row["hgvs_expressions"],
                    "origin_id": row_id
                }

            mutation_disease_dict[(rs_id, disease_id)]["articles"].add(row["citation_id"])

        self.log_info(f"Saving prepared data set to {output_file}")
        with open(output_file, "w", encoding="utf8") as writer:
            head = "\t".join(["source_id", "transcript", "gene", "target_id", "articles", "hgvs", "origin_id"])
            writer.write(head + "\n")

            for key, value in mutation_disease_dict.items():
                doid = "DOID:" + value["target_id"]
                transcript = value["transcript"] if value["transcript"] is not None and type(value["transcript"]) is not float else ""
                articles_str = ",".join(value["articles"])
                hgvs = value["hgvs_expressions"] if value["hgvs_expressions"] is not None else ""
                origin_id = str(value["origin_id"])

                writer.write("\t".join([value["source_id"],
                                        transcript,
                                        value["gene"],
                                        doid,
                                        articles_str,
                                        hgvs,
                                        origin_id
                                        ]))
                writer.write("\n")


if __name__ == "__main__":
    reader = CIViC()
    reader.prepare(CIVIC_EVIDENCES_FILE, CIVIC_VARIANT_SUMMARY_FILE, RSID_TO_HGVS_FILE, CIVIC_DATA_SET_FILE)
