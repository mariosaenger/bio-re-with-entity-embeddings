import multiprocessing
import re
import pandas as pd

from pathlib import Path
from pandas import DataFrame
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from data.disease_ontology import DiseaseOntology
from utils.log_utils import LoggingMixin


TITLE_PATTERN = re.compile("([0-9]+)\\|t\\|(.*)")
ABSTRACT_PATTERN = re.compile("([0-9]+)\\|a\\|(.*)")


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


class Annotation(object):

    def __init__(self, pubmed_id: str, entity_id: str, mention_text: str, start_offset: int, end_offset: int):
        self.pubmed_id = pubmed_id
        self.entity_id = entity_id
        self.mention_text = mention_text
        self.start_offset = start_offset
        self.end_offset = end_offset


class AnnotationExtractor(LoggingMixin):

    def __init__(self):
        super(AnnotationExtractor, self).__init__()

    def extract(self, plain_document: str) -> List[Annotation]:
        title = None
        abstract = None
        annotations = []

        for line in plain_document.split("\n"):
            line = line.strip()
            if not line:
                continue

            title_match = TITLE_PATTERN.match(line)
            if title_match:
                title = title_match.group(2)
                continue

            abstract_match = ABSTRACT_PATTERN.match(line)
            if abstract_match:
                abstract = abstract_match.group(2)
                continue

            annotations += self.parse_annotation_line(line)

        complete_text = title if title else ""
        complete_text = complete_text + " " + abstract if abstract else complete_text
        text_length = len(complete_text)

        # Only keep annotations which are in the title or abstract of the article (not in the full-text!)
        result = []
        for annotation in annotations:
            if annotation.start_offset > text_length:
                continue

            spanned_text = complete_text[annotation.start_offset:annotation.end_offset]
            if spanned_text.lower() == annotation.mention_text.lower():
                result += [annotation]

        return result

    def parse_annotation_line(self, line: str) -> List[Annotation]:
        raise NotImplementedError("Has to be implemented by subclasses!")


class DefaultEntityAnnotationExtractor(AnnotationExtractor):

    def __init__(self, entity_type: str):
        super(DefaultEntityAnnotationExtractor, self).__init__()
        self.entity_type = entity_type.lower()

    def parse_annotation_line(self, line: str) -> List[Annotation]:
        columns = line.split("\t")
        if self.entity_type not in columns[4].lower():
            return []

        if len(columns) < 6:
            #self.log_warn(f"Unexpected line format: {line}")
            return []

        entity_id = self.normalize_id(columns[5])
        if not entity_id:
            return []

        return [
            Annotation(
                pubmed_id=columns[0],
                entity_id=entity_id,
                mention_text=columns[3],
                start_offset=int(columns[1]),
                end_offset=int(columns[2])
            )
        ]

    def normalize_id(self, entity_id: str) -> Optional[str]:
        return entity_id


class ChemicalAnnotationExtractor(DefaultEntityAnnotationExtractor):

    def __init__(self):
        super(ChemicalAnnotationExtractor, self).__init__("Chemical")


class CelllineAnnotationExtractor(DefaultEntityAnnotationExtractor):

    def __init__(self):
        super(CelllineAnnotationExtractor, self).__init__("CellLine")


class DrugAnnotationExtractor(AnnotationExtractor):

    def __init__(self, mesh_to_drugbank: DataFrame):
        super(DrugAnnotationExtractor, self).__init__()
        self.mesh_to_drugbank = mesh_to_drugbank

    def parse_annotation_line(self, line: str) -> List[Annotation]:
        columns = line.split("\t")
        if "Chemical" not in columns[4]:
            return []

        annotations = []

        drugbank_ids = self.get_drugbank_ids_by_mesh(columns[5])
        for drugbank_id in drugbank_ids:
            annotations += [
                Annotation(
                    pubmed_id=columns[0],
                    entity_id=drugbank_id,
                    mention_text=columns[3],
                    start_offset=int(columns[1]),
                    end_offset=int(columns[2])
                )
            ]

        return annotations

    def get_drugbank_ids_by_mesh(self, mesh: str) -> List[str]:
        if mesh in self.mesh_to_drugbank.index:
            return self.mesh_to_drugbank.loc[mesh]["DrugBankIDs"].split("|")

        return []


class DiseaseAnnotationExtractor(AnnotationExtractor):

    def __init__(self, disease_ontology: DiseaseOntology = None):
        super(DiseaseAnnotationExtractor, self).__init__()
        self.disease_ontology = disease_ontology

    def parse_annotation_line(self, line: str) -> List[Annotation]:
        columns = line.split("\t")

        if "Disease" not in columns[4]:
            return []

        if len(columns) < 6:
            #self.log_warn(f"Unexpected line format: {line}")
            return []

        mesh = columns[5]
        disease_ids = [mesh] if not self.disease_ontology else \
            self.disease_ontology.get_doid_by_mesh(mesh)

        annotations = []
        for disease_id in disease_ids:
            annotations += [
                Annotation(
                    pubmed_id=columns[0],
                    entity_id=disease_id,
                    mention_text=columns[3],
                    start_offset=int(columns[1]),
                    end_offset=int(columns[2])
                )
            ]

        return annotations


class GeneAnnotationExtractor(DefaultEntityAnnotationExtractor):

    def __init__(self):
        super(GeneAnnotationExtractor, self).__init__("Gene")

    def normalize_id(self, entity_id: str) -> Optional[str]:
        return "NCBI:" + entity_id


class MutationAnnotationExtractor(DefaultEntityAnnotationExtractor):

    def __init__(self):
        super(MutationAnnotationExtractor, self).__init__("Mutation")

    def normalize_id(self, entity_id: str) -> Optional[str]:
        if not "RS#:" in entity_id:
            return None

        rs_id = None
        for id in entity_id.split(";"):
            if id.startswith("RS#:"):
                rs_id = id.replace("RS#:", "rs")
                break

        return rs_id


class SpeciesAnnotationExtractor(DefaultEntityAnnotationExtractor):

    def __init__(self):
        super(SpeciesAnnotationExtractor, self).__init__("Species")

    def normalize_id(self, entity_id: str) -> Optional[str]:
        return "TAXON:" + entity_id


def extract_annotations(documents: List[str], annotation_extractor: AnnotationExtractor):
    annotations = []
    for plain_document in documents:
        annotations += annotation_extractor.extract(plain_document)

    return annotations

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

            title_match = TITLE_PATTERN.match(line)
            if title_match:
                pubmed_id = title_match.group(1)
                title = title_match.group(2)
                continue

            abstract_match = ABSTRACT_PATTERN.match(line)
            if abstract_match:
                pubmed_id = abstract_match.group(1)
                abstract = abstract_match.group(2)
                continue

        documents[pubmed_id] = Document(pubmed_id, title, abstract)

    return documents


class PubtatorCentral(LoggingMixin):

    def __init__(self):
        super(PubtatorCentral, self).__init__()

    def extract_entity_annotations(self, offsets_file: Path, extractor: AnnotationExtractor) -> Tuple[DataFrame, DataFrame]:
        plain_documents = self.read_plain_documents(offsets_file)

        annotations = self.extract_annotations_parallel(
            plain_documents=plain_documents,
            annotation_extractor=extractor,
            processes=16,
            batch_size=2000
        )

        return self.build_mappings(annotations)

    def read_plain_documents(self, offsets_file: Path) -> List[str]:
        self.log_info(f"Reading plain documents from {offsets_file}")

        documents = list()
        with open(str(offsets_file), "r", encoding="utf-8") as input_reader:
            document = None
            for line in input_reader:
                line = line.strip()

                if not line:
                    if document:
                        documents.append(document)
                        document = None

                if not document:
                    document = ""

                document = document + "\n" + line

        return documents

    def extract_annotations_parallel(self, plain_documents: List[str], annotation_extractor: AnnotationExtractor,
                                     processes: int, batch_size: int) -> List[Annotation]:
        self.log_info(f"Start extracting annotations from {len(plain_documents)} documents "
                      f"(processes={processes}|batch-size={batch_size})")

        annotations = []
        with multiprocessing.Pool(processes) as pool:
            num_batches = (len(plain_documents) - 1) // batch_size + 1
            futures = []

            self.log_info(f"Submitting annotation extraction jobs")
            for i in tqdm(range(0, len(plain_documents), batch_size), total=num_batches):
                document_batch = plain_documents[i:i + batch_size]
                future = pool.apply_async(extract_annotations, [document_batch, annotation_extractor])
                futures.append(future)

            self.log_info("Collecting results")
            for future in tqdm(futures, desc="collect-result", total=len(futures)):
                annotations += future.get()

        self.log_info(f"Found {len(annotations)} annotations in total")

        return annotations

    def build_mappings(self, annotations: List[Annotation]) -> Tuple[DataFrame, DataFrame]:
        pubmed2entity = {}
        entity2pubmed = {}

        for annotation in annotations:
            pubmed_id = annotation.pubmed_id
            entity_id = annotation.entity_id

            if not pubmed_id in pubmed2entity:
                pubmed2entity[pubmed_id] = {
                    "entity_ids": {entity_id}
                }
            else:
                pubmed2entity[pubmed_id]["entity_ids"].add(entity_id)

            if not entity_id in entity2pubmed:
                entity2pubmed[entity_id] = {
                    "articles": {pubmed_id}
                }
            else:
                entity2pubmed[entity_id]["articles"].add(pubmed_id)

        pubmed2entity = pd.DataFrame.from_dict(pubmed2entity, orient="index")
        entity2pubmed = pd.DataFrame.from_dict(entity2pubmed, orient="index")

        return pubmed2entity, entity2pubmed

    def parse_raw_documents_parallel(self, raw_documents: List[str], threads: int,
                                     batch_size: int) -> Dict[str, Document]:
        self.log_info(f"Start parsing {len(raw_documents)} documents (threads={threads}|batch-size={batch_size})")

        parsed_documents = dict()
        with multiprocessing.Pool(threads) as pool:
            num_batches = (len(raw_documents) - 1) // batch_size + 1
            self.log_info(f"Creating jobs to parse the documents")

            futures = []

            for i in tqdm(range(0, len(raw_documents), batch_size), desc="create-jobs",total=num_batches):
                document_batch = raw_documents[i:i + batch_size]
                future = pool.apply_async(parse_raw_documents, document_batch)
                futures.append(future)

            self.log_info("Collecting parse results")
            for future in tqdm(futures, desc="collect-result", total=len(futures)):
                parsed_documents.update(future.result())

        return parsed_documents
