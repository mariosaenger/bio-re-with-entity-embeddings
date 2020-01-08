import argparse
import os
import requests
import time
import xml.etree.ElementTree as ET

from typing import List, Dict, Set, Tuple
from tqdm import tqdm

from utils.log_utils import LoggingMixin


class RsidToHgvsMapper(LoggingMixin):

    def __init__(self):
        super(RsidToHgvsMapper, self).__init__(self.__class__.__name__)

    def retrieve_hgvs_expressions(self, rs_ids: List[str], existing_mappings: Dict[str, Set[str]],
                                  batch_size: int = 10, sleep_time_sec : float = 0.5) -> Tuple[Dict[str, Set[str]], List[str]]:
        self.log_info(f"Start retrieving hgvs expressions for {len(rs_ids)} rs ids")
        rsid_to_hgvs_map = dict()
        rsid_to_hgvs_map.update(existing_mappings)

        rs_ids_to_crawl = list(set(rs_ids).difference(set(rsid_to_hgvs_map.keys())))
        self.log_info(f"Found {len(rs_ids_to_crawl)} rs ids not already crawled")

        with open("rsid.tsv", "a", encoding="utf8") as temp_writer:
            number_of_batches = int(len(rs_ids_to_crawl) / batch_size)
            for i in tqdm(range(0, len(rs_ids_to_crawl), batch_size), desc="Retrieve hgvs expressions", total=number_of_batches):
                batch_rs_ids = ",".join(rs_ids_to_crawl[i:i+batch_size])

                try:
                    get_request_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=snp&id={batch_rs_ids}&retmode=xml"
                    response = requests.get(get_request_url)

                    if response.status_code == 200:
                        root = ET.fromstring(response.text)

                        for rs_element in root.iter("{https://www.ncbi.nlm.nih.gov/SNP/docsum}Rs"):
                            rs_id = "rs" + str(rs_element.attrib["rsId"])

                            if rs_id in rsid_to_hgvs_map:
                                self.log_warn(f"Found duplicate rs id {rs_id}")
                                continue

                            hgvs_expressions = set([hgvs.text for hgvs in rs_element.iter("{https://www.ncbi.nlm.nih.gov/SNP/docsum}hgvs")])
                            rsid_to_hgvs_map[rs_id] = hgvs_expressions

                            hgvs_str = ";".join(hgvs_expressions)
                            temp_writer.write(f"{rs_id}\t{hgvs_str}\n")
                            temp_writer.flush()
                    else:
                        self.logger.warn(f"Request {get_request_url} responded with status code {response.status_code}")

                except requests.exceptions.HTTPError as e:
                    self.log_warn(f"Request {get_request_url} failed with HTTPError: {e}")
                except requests.exceptions.ConnectionError as e:
                    self.log_warn(f"Request {get_request_url} failed with ConnectionError: {e}")
                except requests.exceptions.Timeout as e:
                    self.log_warn(f"Request {get_request_url} failed with Timeout: {e}")
                except requests.exceptions.RequestException as e:
                    self.log_warn(f"Request {get_request_url} failed with RequestException: {e}")

                time.sleep(sleep_time_sec)

        self.log_info(f"Finished retrieval. Found hgvs expressions for {len(rsid_to_hgvs)} rs ids")
        missing_rs_ids = list(set(rs_ids).difference(set(rsid_to_hgvs_map.keys())))

        return rsid_to_hgvs_map, missing_rs_ids

    def save_mapping(self, rsid_to_hgvs: Dict[str, Set[str]], output_file: str):
        self.log_info(f"Saving {len(rsid_to_hgvs)} entries to {output_file}")
        with open(output_file, "w", encoding="utf8") as writer:
            for key, value in rsid_to_hgvs.items():
                value_str = ";".join(value)
                writer.write(f"{key}\t{value_str}\n")

            writer.close()

    def read_mapping(self, input_file: str) -> Dict[str, Set[str]]:
        self.log_info(f"Loading rsid to hgvs expresssion mappings from {input_file}")
        rsid_to_hgvs = dict()
        with open(input_file, "r", encoding="utf8") as reader:
            for line in reader.readlines():
                columns = line.strip().split()
                rsid_to_hgvs[columns[0]] = set(columns[1].split(";"))

        self.log_info(f"Found {len(rsid_to_hgvs)} mappings")
        return rsid_to_hgvs

    def save_rs_id_list(self, rs_ids : List[str], output_file: str) -> None:
        self.log_info(f"Saving {len(rs_ids)} to {output_file}")
        with open(output_file, "w", encoding="utf8") as writer:
            writer.write("\n".join([f"{id}" for id in rs_ids]))
            writer.close()

    def read_rs_id_list(self, input_file: str) -> List[str]:
        self.log_info(f"Loading rs id list from {input_file}")
        rs_ids = []
        with open(input_file, "r", encoding="utf8") as reader:
            rs_ids = [line.strip( )for line in reader.readlines()]
            reader.close()

        self.log_info(f"Found {len(rs_ids)} rs identifier")
        return rs_ids

    def read_hgvs_to_rsid_mapping(self, input_file: str) -> Dict[str, str]:
        self.log_info(f"Read rsid <-> hgvs expression mapping from {input_file}")
        hgvs_to_rsid = dict()

        with open(input_file, "r", encoding="utf8") as reader:
            for line in reader.readlines():
                rsid, hgvs_expressions = line.strip().split("\t")
                for hgvs in hgvs_expressions.split(";"):
                    hgvs, change = hgvs.split(":")
                    hgvs = hgvs.split(".")[0] + ":" + change
                    if hgvs in hgvs_to_rsid:
                        #self.log_warn(f"Found duplicate hgvs mapping for {hgvs}: {rsid} and {hgvs_to_rsid[hgvs]}")
                        continue
                    else:
                        hgvs_to_rsid[hgvs] = rsid

        return hgvs_to_rsid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_list", required=True, help="Path to the input rs id list")
    parser.add_argument("--output_file", required=True, help="Path to the output file")
    parser.add_argument("--error_list_file", required=True, help="Path to the file for storing the error ids")

    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--sleep_time", default=0.01, type=float)

    args = parser.parse_args()

    mapper = RsidToHgvsMapper()
    input_rs_id_list = mapper.read_rs_id_list(args.input_list)

    if os.path.exists(args.output_file):
        rsid_to_hgvs = mapper.read_mapping(args.output_file)
    else:
        rsid_to_hgvs = {}

    rsid_to_hgvs, error_ids = mapper.retrieve_hgvs_expressions(
        input_rs_id_list, rsid_to_hgvs, args.batch_size, args.sleep_time)
    mapper.save_mapping(rsid_to_hgvs, args.output_file)
    mapper.save_rs_id_list(error_ids, args.error_list_file)
