import pandas as pd

from lxml import etree
from lxml.etree import XMLParser
from pandas import DataFrame
from tqdm import tqdm

from utils.log_utils import LoggingMixin


class DrugBankReader(LoggingMixin):

    def __init__(self):
        super(DrugBankReader, self).__init__()

    def read(self, database_file: str) -> DataFrame:
        self.log_info("Start reading DrugBank database from %s", database_file)

        xml_parser = XMLParser(encoding="utf-8", huge_tree=True, ns_clean=True)
        tree = etree.parse(database_file, xml_parser)
        root = tree.getroot()

        dd_interactions = dict()

        drug_elements = root.findall("{http://www.drugbank.ca}drug")
        for drug_element in tqdm(drug_elements, total=len(drug_elements)):
            primary_id = None

            id_elements = drug_element.findall("{http://www.drugbank.ca}drugbank-id")
            for id_element in id_elements:
                if "primary" in id_element.attrib:
                    primary_id = id_element.text
                    continue

            if primary_id is None:
                self.log_warn("Can't find primary id for %s", drug_element)
                continue

            if primary_id in dd_interactions:
                self.log_warn("Found multiple entries for drug id %s", primary_id)
                continue

            drug_name = drug_element.find("{http://www.drugbank.ca}name").text

            drug_interactions_element = drug_element.find("{http://www.drugbank.ca}drug-interactions")
            drug_interaction_elements = drug_interactions_element.findall("{http://www.drugbank.ca}drug-interaction")

            drug_interactions = set()
            for drug_interaction_element in drug_interaction_elements:
                other_id_element = drug_interaction_element.find("{http://www.drugbank.ca}drugbank-id")
                drug_interactions.add(other_id_element.text)

            drug_interactions = drug_interactions if len(drug_interactions) > 0 else None
            dd_interactions[primary_id] = {"drug_id" : primary_id,
                                           "name": drug_name,
                                           "interactions" : drug_interactions}

        drugbank_dataset = pd.DataFrame.from_dict(dd_interactions, orient="index")
        self.log_info("Finished loading of DrugBank. Found %s drug entries.", len(drugbank_dataset))

        return drugbank_dataset


