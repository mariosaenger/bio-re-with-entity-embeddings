import argparse
import logging
import pandas as pd

from pathlib import Path
from sklearn.pipeline import Pipeline

from data.disease_ontology import DiseaseOntology
from data.pubtator import (
    AnnotationExtractor,
    PubtatorCentral,
    MutationAnnotationExtractor,
    DrugAnnotationExtractor,
    DiseaseAnnotationExtractor, CelllineAnnotationExtractor, ChemicalAnnotationExtractor, GeneAnnotationExtractor,
    SpeciesAnnotationExtractor
)
from data.resource_handler import ResourceHandler
from extract_articles import PubtatorArticleExtractor
from prepare_doc2vec_input import Doc2VecPreparation
from utils.log_utils import LogUtil, LoggingMixin
from utils.pandas_utils import PandasUtil as pdu


class EntityDataSetPreparation(LoggingMixin):

    def __init__(self):
        super(EntityDataSetPreparation, self).__init__()

    def run(self, working_directory: Path, pubtator_offset_file: Path,
            entity_type: str, annotation_extractor: AnnotationExtractor):
        self.log_info(f"Start preparation of PubMed {entity_type} occurrences")

        entity_ds_dir = working_directory / entity_type
        entity_ds_dir.mkdir(parents=True, exist_ok=True)

        self.log_info(f"Start extraction of mutation annotations from {pubtator_offset_file}")
        pubtator_central = PubtatorCentral()
        pubmed2entity, entity2pubmed = pubtator_central.extract_entity_annotations(
            offsets_file=pubtator_offset_file,
            extractor=annotation_extractor
        )
        self.log_info(f"Found {len(entity2pubmed)} distinct entities")

        instances_file = entity_ds_dir / "instances.tsv"
        pubmed_ids_file = entity_ds_dir / "pubmed_ids.txt"

        self.log_info("Preparing and saving entity-to-PubMed information")
        pipeline = Pipeline([
            (
                "ConvertArticlesToString",
                pdu.map("articles", self.set_to_string, "articles_str")
            ),
            (
                "ExtractPubMedIds",
                pdu.extract_unique_values("articles", pubmed_ids_file)
            )
        ])

        entity2pubmed = pipeline.fit_transform(entity2pubmed)
        entity2pubmed.to_csv(instances_file, sep="\t", columns=["articles_str"], index_label="entity_id")

        self.log_info("Preparing and saving PubMed-to-entity information")
        pipeline = Pipeline([
            (
                "ConvertEntityIdsToString",
                pdu.map("entity_ids", self.set_to_string, "entity_ids_str")
            ),
        ])
        pubmed2entity = pipeline.fit_transform(pubmed2entity)

        pubmed2entity_file = entity_ds_dir / "pubmed2entity.tsv"
        pubmed2entity.to_csv(pubmed2entity_file,  sep="\t", columns=["entity_ids_str"], index_label="pubmed_id")

    @staticmethod
    def set_to_string(values):
        if len(values) == 0:
            return None

        return ";;;".join([str(value) for value in sorted(values)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", type=str, required=True,
                        help="Directory which will hold all (intermediate) output files")
    parser.add_argument("--entity_type", type=str, required=True,
                        choices=["chemical", "cellline", "drug", "disease", "disease-doid",
                                 "gene", "mutation", "species"],
                        help="Target entity type (drug, disease, or mutation)")

    # Optional parameters
    parser.add_argument("--resource_dir", type=str, required=False, default="_resources",
                        help="Path to the directory containing the resources")
    parser.add_argument("--use_caching", type=bool, required=False, default=True,
                        help="Indicate whether to perform all preparation steps from scratch or use "
                             "existing results from previous executions.")
    args = parser.parse_args()

    logger = LogUtil.create_logger("prepare_pair_dataset", logging.INFO)
    logger.info(f"Start preparation of {args.entity_type} data set")

    use_caching = args.use_caching

    resource_dir = Path(args.resource_dir)
    resources = ResourceHandler(resource_dir)

    working_dir = Path(args.working_dir)
    entity_type_dir = working_dir / args.entity_type

    instance_file = entity_type_dir / "instances.tsv"
    pubmed_ids_file = entity_type_dir / "pubmed_ids.txt"
    pubmed2entity_file = entity_type_dir / "pubmed2entity.tsv"

    # Run preparation of the data set for the given entity type using the information
    # provided by the PubTator files
    if (
            use_caching is False
            or
            not (
                instance_file.exists() and
                pubmed_ids_file.exists() and
                pubmed2entity_file.exists()
            )
    ):
        extractor = None

        if args.entity_type == "cellline":
            extractor = CelllineAnnotationExtractor()
        elif args.entity_type == "chemical":
            extractor = ChemicalAnnotationExtractor()
        elif args.entity_type == "disease":
            extractor = DiseaseAnnotationExtractor()
        elif args.entity_type == "disease-doid":
            disease_ontology = DiseaseOntology(resources.get_disease_ontology_tsv_file())
            extractor = DiseaseAnnotationExtractor(disease_ontology)
        elif args.entity_type == "drug":
            mesh_to_drugbank_mapping = pd.read_csv(resources.get_mesh_to_drugbank_file(), sep="\t")
            extractor = DrugAnnotationExtractor(mesh_to_drugbank_mapping)
        elif args.entity_type == "gene":
            extractor = GeneAnnotationExtractor()
        elif args.entity_type == "mutation":
            extractor = MutationAnnotationExtractor()
        elif args.entity_type == "species":
            extractor = SpeciesAnnotationExtractor()
        else:
            raise NotImplementedError(f"Unsupported entity type {args.entity_type}")

        entity_preparation = EntityDataSetPreparation()
        entity_preparation.run(
            working_directory=working_dir,
            pubtator_offset_file=resources.get_pubtator_offset_file(),
            entity_type=args.entity_type,
            annotation_extractor=extractor
        )

        # Skipping caching for all following steps to prevent unintended caching issues
        use_caching = False

    else:
        logger.info("Skipping PubTator preparation")

    # Extract the necessary articles from the PubTator offset file
    articles_file = entity_type_dir / "articles.txt"
    if (
            use_caching is False
            or
            not articles_file.exists()
    ):
        extractor = PubtatorArticleExtractor()
        extractor.run(
            offset_file=resources.get_pubtator_offset_file(),
            pubmed_ids_file=pubmed_ids_file,
            output_file=articles_file,
            threads=16,
            batch_size=2000
        )

        # Skipping caching for all following steps to prevent unintended caching issues
        use_caching = False
    else:
        logger.info("Skipping extraction of articles")

    # Create the input files for the Doc2Vec entity embedding learning
    doc2vec_input_file = entity_type_dir / "doc2vec_input.txt"
    if (
        use_caching is False
        or
        not doc2vec_input_file.exists()
    ):
        extractor = Doc2VecPreparation()
        extractor.run(
            input_file=instance_file,
            id_columns=["entity_id"],
            article_file=articles_file,
            article_column="articles_str",
            output_file=doc2vec_input_file
        )

    else:
        logger.info("Skipping preparation of the Doc2Vec input")

    logger.info(f"Finished preparation of {args.entity_type} data set")
