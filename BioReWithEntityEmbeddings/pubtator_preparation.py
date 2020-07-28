import os
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from data.disease_ontology import DiseaseOntology
from data.pubtator import PubtatorPreparationUtils, Pubtator
from data.resources import PUBTATOR_MUT2PUB_FILE, MESH_TO_DRUGBANK_MAPPING, PUBTATOR_CHEM2PUB_FILE, DO_ONTOLOGY_FILE, \
    PUBTATOR_DIS2PUB_FILE
from utils.log_utils import LoggingMixin
from utils.pandas_utils import PandasUtil as pdu, MeshTermToDoidMapper, MeshTermToDrugBankIdMapper


class EntityDataSetPreparation(LoggingMixin):

    def __init__(self):
        super(EntityDataSetPreparation, self).__init__()

    def run(self, working_dir: Path):
        raise NotImplementedError("run() has to be implemented by sub-classes")


class PubtatorMutationDataSetPreparation(EntityDataSetPreparation):

    def __init__(self):
        super(PubtatorMutationDataSetPreparation, self).__init__()

    def run(self, working_directory: Path):
        self.log_info("Start preparation of PubMed mutation occurrences")

        entity_ds_dir = working_directory / "mutation"
        os.makedirs(str(entity_ds_dir), exist_ok=True)

        pubtator = Pubtator()

        self.log_info("Read mutation data from %s", PUBTATOR_MUT2PUB_FILE)
        mutation_overview = pubtator.build_mutation_overview(PUBTATOR_MUT2PUB_FILE)
        self.log_info("Found %s mutations in total", len(mutation_overview))

        mutation_instances_file = entity_ds_dir / "instances.tsv"
        mutation_pubmed_ids_file = entity_ds_dir / "pubmed_ids.txt"

        self.log_info("Saving mutation information to entity data set directory")
        mutation_pipeline = Pipeline([
            (
                "ConvertArticlesToString",
                pdu.map("articles", PubtatorPreparationUtils.set_to_string, "articles_str")
            ),
            (
                "ExtractPubMedIds",
                pdu.extract_unique_values("articles", mutation_pubmed_ids_file)
            ),
            (
                "RenameIdColumn",
                pdu.rename_columns({"rs_identifier": "entity_id"})
            )
        ])

        mutation_overview = mutation_pipeline.fit_transform(mutation_overview)
        mutation_overview.to_csv(mutation_instances_file, sep="\t",
                                 columns=["entity_id", "articles_str"], index=False)

        self.log_info("Create mapping from PubMed id to mutation id")
        pubmed_to_mutation = PubtatorPreparationUtils.create_pubmed_id_to_entity_map(mutation_overview)
        pipeline = Pipeline([
            (
                "ConvertEntityIdsToString",
                pdu.map("entity_ids", PubtatorPreparationUtils.set_to_string, "entity_ids_str")
            ),
        ])
        pubmed_to_mutation = pipeline.fit_transform(pubmed_to_mutation)

        pubmed2entity_file = entity_ds_dir / "pubmed2entity.tsv"
        pubmed_to_mutation.to_csv(pubmed2entity_file,  sep="\t",
                                  columns=["entity_ids_str"], index_label="pubmed_id")


class PubtatorDrugOccurrencesPreparation(EntityDataSetPreparation):

    def __init__(self):
        super(PubtatorDrugOccurrencesPreparation, self).__init__()

    def run(self, working_directory: Path):
        self.log_info("Start preparation of drug occurrences")

        entity_ds_dir = working_directory / "drug"
        os.makedirs(str(entity_ds_dir), exist_ok=True)

        pubtator = Pubtator()

        mesh_to_drugbank_mapping = pd.read_csv(MESH_TO_DRUGBANK_MAPPING, sep="\t", )

        # Drug data
        self.log_info("Read disease data from %s", PUBTATOR_CHEM2PUB_FILE)
        drug_data = pubtator.build_drug_overview(PUBTATOR_CHEM2PUB_FILE)
        self.log_info("Found %s unique diseases in total", len(drug_data))

        self.log_info("Saving disease information to entity data set directory")
        drug_pubmed_ids_file = entity_ds_dir / "pubmed_ids.txt"

        pipeline = Pipeline([
            (
                "MapMashTermToDrugBankId",
                MeshTermToDrugBankIdMapper(mesh_to_drugbank_mapping)
            ),
            (
                "ConvertArticlesToString",
                pdu.map("articles", PubtatorPreparationUtils.set_to_string, "articles_str")
            ),
            (
                "RenameIdColumn",
                pdu.rename_columns({"MeshID": "entity_id"})
            ),
            (
                "ExtractPubMedIds",
                pdu.extract_unique_values("articles", drug_pubmed_ids_file)
            ),
        ])
        drug_data = pipeline.fit_transform(drug_data)

        drug_instances_file = entity_ds_dir / "instances.tsv"
        drug_data.to_csv(drug_instances_file, sep="\t", columns=["entity_id", "articles_str"], index=False)

        self.log_info("Create mapping from PubMed id to Drug id")
        pubmed_to_drug = PubtatorPreparationUtils.create_pubmed_id_to_entity_map(drug_data)
        pipeline = Pipeline([
            ("ConvertEntityIdsToString",
             pdu.map("entity_ids", PubtatorPreparationUtils.set_to_string, "entity_ids_str")),
        ])
        pubmed_to_drug = pipeline.fit_transform(pubmed_to_drug)

        pubmed2entity_file = entity_ds_dir / "pubmed2entity.tsv"
        pubmed_to_drug.to_csv(pubmed2entity_file, sep="\t", columns=["entity_ids_str"], index_label="pubmed_id")


class PubtatorDiseaseOccurrencesPreparation(EntityDataSetPreparation):

    def __init__(self):
        super(PubtatorDiseaseOccurrencesPreparation, self).__init__()

    def run(self, working_directory: Path):
        self.log_info("Start preparation of disease occurrences")

        entity_ds_dir = working_directory / "disease"
        os.makedirs(str(entity_ds_dir), exist_ok=True)

        self.disease_ontology = DiseaseOntology(DO_ONTOLOGY_FILE)
        pubtator = Pubtator()

        # Disease data
        self.log_info("Read disease data from %s", PUBTATOR_DIS2PUB_FILE)
        disease_overview = pubtator.build_disease_overview(PUBTATOR_DIS2PUB_FILE)
        self.log_info("Found %s unique diseases in total", len(disease_overview))

        disease_pubmed_ids_file = entity_ds_dir / "pubmed_ids.txt"

        self.log_info("Saving disease information to entity data set directory")
        disease_pipeline = Pipeline([
            (
                "MapMashTermTodoDoid",
                MeshTermToDoidMapper(self.disease_ontology)
            ),
            (
                "FilterEntriesWithoutDoid",
                pdu.not_null("doid")
            ),
            (
                "ConvertArticlesToString",
                pdu.map("articles", PubtatorPreparationUtils.set_to_string, "articles_str")
            ),
            (
                "RenameIdColumn",
                pdu.rename_columns({"doid": "entity_id"})
            ),
            (
                "ExtractPubMedIds",
                pdu.extract_unique_values("articles", disease_pubmed_ids_file)
            ),
        ])

        disease_overview = disease_pipeline.fit_transform(disease_overview)

        disease_instances_file = entity_ds_dir / "instances.tsv"
        disease_overview.to_csv(disease_instances_file, sep="\t", columns=["entity_id", "articles_str"], index=False)

        self.log_info("Create mapping from PubMed id to disease id")
        pubmed_to_disease = PubtatorPreparationUtils.create_pubmed_id_to_entity_map(disease_overview)
        pipeline = Pipeline([
            (
                "ConvertEntityIdsToString",
                pdu.map("entity_ids", PubtatorPreparationUtils.set_to_string, "entity_ids_str")
            ),
        ])
        pubmed_to_disease = pipeline.fit_transform(pubmed_to_disease)

        pubmed2entity_file = entity_ds_dir / "pubmed2entity.tsv"
        pubmed_to_disease.to_csv(pubmed2entity_file,  sep="\t", columns=["entity_ids_str"], index_label="pubmed_id")


class CooccurrencePreparator(LoggingMixin):

    def __init__(self):
        super(CooccurrencePreparator, self).__init__()

    def run(self, working_directory: Path, source_type: str, target_type: str):
        self.log_info(f"Start preparing {source_type}-{target_type} co-occurrences")

        source_mapping = self.read_pubmed2entity(working_directory, source_type)
        target_mapping = self.read_pubmed2entity(working_directory, target_type)

        join_result = source_mapping.join(target_mapping, lsuffix="_source", rsuffix="_target")
        join_result = join_result[join_result["entity_ids_str_target"].notnull()]
        join_result = join_result[join_result["entity_ids_str_target"].notna()]
        self.log_info(f"Found {len(join_result)} documents with co-occurrences")

        pubmed2pairs = {}
        pair2pubmed = {}

        for pubmed_id, row in tqdm(join_result.iterrows(), total=len(join_result)):
            source_ids = row["entity_ids_str_source"].split(";;;")
            target_ids = row["entity_ids_str_target"].split(";;;")

            pair_ids = [(s_id, t_id) for s_id in source_ids for t_id in target_ids]
            pubmed2pairs[str(pubmed_id)] = {
                "pair_ids_str": ";;;".join(["##".join(list(p_id)) for p_id in pair_ids])
            }

            for pair_id in pair_ids:
                if pair_id in pair2pubmed:
                    pair_entry = pair2pubmed[pair_id]
                    pair_entry["pubmed_ids_str"] += ";;;" + str(pubmed_id)
                else:
                    pair_entry = {
                        "source_id": pair_id[0],
                        "target_id": pair_id[1],
                        "pubmed_ids_str": str(pubmed_id)
                    }

                pair2pubmed[pair_id] = pair_entry

        pair_output_dir = working_directory / f"{source_type}-{target_type}"
        os.makedirs(str(pair_output_dir), exist_ok=True)

        pubmed2pair = pd.DataFrame.from_dict(pubmed2pairs, orient="index")
        pubmed2pair_file = pair_output_dir / "pubmed2pair.tsv"
        pubmed2pair.to_csv(pubmed2pair_file, sep="\t", index_label="pubmed_id")

        pubmed_id_file = pair_output_dir / "pubmed_ids.txt"
        with open(str(pubmed_id_file), "w") as writer:
            writer.write("\n".join([str(id) for id in pubmed2pair.index.values]))

        pair2pubmed = pd.DataFrame.from_dict(pair2pubmed, orient="index")
        pair2pubmed_file = pair_output_dir / "instances.tsv"
        pair2pubmed.to_csv(pair2pubmed_file, sep="\t", index=False)

    def read_pubmed2entity(self, working_directory: Path, entity_type) -> pd.DataFrame:
        input_file = working_directory / entity_type / "pubmed2entity.tsv"

        self.log_info(f"Reading PubMed2Entity mapping from {input_file}")
        mapping = pd.read_csv(input_file, sep="\t", index_col="pubmed_id")
        self.log_info(f"Found {len(mapping)} mapping entries")

        return mapping
