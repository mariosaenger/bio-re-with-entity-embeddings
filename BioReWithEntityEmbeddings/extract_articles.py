import argparse
import multiprocessing
import time

from pathlib import Path
from typing import List
from tqdm import tqdm

from data.pubtator import PubtatorCentral, TITLE_PATTERN, ABSTRACT_PATTERN
from data.resource_handler import ResourceHandler
from utils.log_utils import LoggingMixin


def filter_by_pubmed_ids(raw_documents: List[str], pubmed_ids: dict) -> List[str]:
    matching_documents = list()
    for raw_document in raw_documents:
        document_lines = raw_document.split("\n")
        pubmed_id = None
        document = None

        for line in document_lines:
            line = line.strip()

            title_match = TITLE_PATTERN.match(line)
            if title_match:
                pubmed_id = title_match.group(1).strip()
                if not document:
                    document = line
                else:
                    document += "\n" + line
                
                continue

            abstract_match = ABSTRACT_PATTERN.match(line)
            if abstract_match:
                pubmed_id = abstract_match.group(1).strip()
                if not document:
                    document = line
                else:
                    document += "\n" + line

                continue

            if line:
                break

        if pubmed_id is not None and pubmed_id in pubmed_ids and document:
            document += "\n\n"
            matching_documents.append(document)

    return matching_documents


class PubtatorArticleExtractor(LoggingMixin):

    def __init__(self):
        super(PubtatorArticleExtractor, self).__init__()

    def run(self, offset_file: Path, pubmed_ids_file: Path, output_file: Path, processes: int, batch_size: int):
        self.log_info(f"Start extraction of PubTator documents from {offset_file}")

        self.log_info(f"Reading PubMed identifiers from {pubmed_ids_file}")
        with open(str(pubmed_ids_file), "r", encoding="utf-8") as pubmed_id_reader:
            pubmed_ids = dict([(line.strip(), 1) for line in pubmed_id_reader.readlines()])
            pubmed_id_reader.close()

        self.log_info(f"Found {len(pubmed_ids)} distinct PubMed identifiers")

        pubtator = PubtatorCentral()

        self.log_info(f"Start reading documents from {offset_file}")
        documents = pubtator.read_plain_documents(offset_file)
        self.log_info("Found %s documents in total", len(documents))

        self.logger.info(f"Creating article extraction jobs (threads={processes} | batch-size={batch_size})")
        num_batches = (len(documents) - 1) // batch_size + 1
        pool = multiprocessing.Pool(processes)

        futures = []

        for i in tqdm(range(0, len(documents), batch_size), total=num_batches):
            document_batch = documents[i:i + batch_size]
            future = pool.apply_async(filter_by_pubmed_ids, [document_batch, pubmed_ids])
            futures.append(future)

        self.log_info("Submitted all tasks!")
        pool.close()

        self.log_info(f"Filtering documents")
        # all_documents = []
        # unfinished_futures = [future for future in futures]
        # progress = tqdm(total=len(futures))
        #
        # while len(unfinished_futures) > 0:
        #     still_unfinished_futures = []
        #     for future in unfinished_futures:
        #         if future.ready():
        #             all_documents += [future.get()]
        #             progress.update(1)
        #         else:
        #             still_unfinished_futures.append(future)
        #
        #     time.sleep(5)
        #     unfinished_futures = still_unfinished_futures
        #
        # self.log_info(f"Writing articles to {output_file}")
        # with open(str(output_file), "w", encoding="utf-8") as output_writer:
        #     for document in tqdm(all_documents, total=len(all_documents)):
        #         output_writer.write("".join(document))
        #
        #     output_writer.close()

        with open(str(output_file), "w", encoding="utf-8") as output_writer:
            for future in tqdm(futures, total=len(futures)):
                documents = future.get()
                output_writer.write("".join(documents))
                del documents

            output_writer.close()

        pool.join()
        self.log_info("Finished extraction of documents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="PubTator article extractor")
    parser.add_argument("--pubmed_id_file", type=str, required=True,
                        help="Path to the file containing a list of pubmed ids")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to the output file")

    # Optional parameters
    parser.add_argument("--resource_dir", type=str, required=False, default="_resources",
                        help="Path to the directory containing the resources")
    parser.add_argument("--processes", type=int, required=False, default=16,
                        help="Number of threads to use",)
    parser.add_argument("--batch_size", type=int, required=False, default=2000,
                        help="Number of documents per job")
    args = parser.parse_args()

    resource_dir = Path(args.resource_dir)
    resources = ResourceHandler(resource_dir)

    extractor = PubtatorArticleExtractor()
    extractor.run(
        offset_file=resources.get_pubtator_offset_file(),
        pubmed_ids_file=Path(args.pubmed_id_file),
        output_file=Path(args.output_file),
        processes=args.processes,
        batch_size=args.batch_size
    )
