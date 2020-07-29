from pathlib import Path


class ResourceHandler(object):

    def __init__(self, resource_dir: Path):
        self.resource_dir = resource_dir

    def get_pubtator_offset_file(self) -> Path:
        return self.resource_dir / "pubtator_central" / "bioconcepts2pubtatorcentral.offset"

    def get_disease_ontology_obo_file(self) -> Path:
        return self.resource_dir / "do" / "doid.obo"

    def get_disease_ontology_tsv_file(self) -> Path:
        return self.resource_dir / "do" / "doid.tsv"

    def get_mesh_to_drugbank_file(self) -> Path:
        return Path("../resources/mappings/drug_mapping.tsv")
