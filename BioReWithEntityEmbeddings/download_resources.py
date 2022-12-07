import argparse
import re
import tempfile
import os
import shutil
import urllib.request as request

from pathlib import Path
from tqdm import tqdm

from data.disease_ontology import DiseaseOntologyHandler
from data.resource_handler import ResourceHandler
from utils.log_utils import LoggingMixin


class ResourceDownloader(LoggingMixin):

    def download_and_unzip_archive(self, archive_url: str, resource_dir: Path):
        file_name = self.get_plain_archive_name(archive_url)

        file_path = resource_dir / file_name
        if file_path.exists():
            self.log_info(f"Archive {file_name} already in cache!")
            return

        self.log_info(f"Downloading archive {archive_url}")
        cached_archive = self.cache_resource(archive_url, resource_dir)

        self.log_info(f"Extracting archive {cached_archive}")
        self.unpack_file(cached_archive, resource_dir / file_name, keep=False)

    def download_file(self, file_url: str, cache_dir: Path):
        self.cache_resource(file_url, cache_dir)

    def cache_resource(self, resource_url: str, cache_dir: Path):
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = re.sub(r".+/", "", resource_url)

        cache_path = cache_dir / filename
        if cache_path.exists():
            return cache_path

        fd, temp_filename = tempfile.mkstemp()
        self.log_info(f"{resource_url} not found in cache, downloading to {temp_filename}")

        # GET file object
        response = request.urlopen(resource_url)
        content_length = int(response.headers["content-length"])
        block_size = 2048

        if content_length:
            length = int(content_length)
            block_size = min(2048, length // 100)

        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total)

        with open(temp_filename, "wb") as writer:
            while True:
                block = response.read(block_size)
                if not block:
                    break
                writer.write(block)
                progress.update(len(block))

        progress.close()

        self.log_info(f"Copying {temp_filename} to cache at {cache_path}")
        shutil.copyfile(temp_filename, str(cache_path))
        self.log_info(f"Removing temp file {temp_filename}")

        os.close(fd)
        os.remove(temp_filename)

        return cache_path

    def unpack_file(self, file: Path, unpack_to: Path, mode: str = None, keep: bool = True):
        if mode == "zip" or (mode is None and str(file).endswith("zip")):
            from zipfile import ZipFile

            with ZipFile(file, "r") as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(unpack_to)

        elif mode == "targz" or (
                mode is None and str(file).endswith("tar.gz") or str(file).endswith("tgz")
        ):
            import tarfile

            with tarfile.open(file, "r:gz") as tarObj:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tarObj, unpack_to)

        elif mode == "tar" or (mode is None and str(file).endswith("tar")):
            import tarfile

            with tarfile.open(file, "r") as tarObj:
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tarObj, unpack_to)

        elif mode == "gz" or (mode is None and str(file).endswith("gz")):
            import gzip

            with gzip.open(str(file), "rb") as f_in:
                with open(str(unpack_to), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        else:
            if mode is None:
                raise AssertionError(f"Can't infer archive type from {file}")
            else:
                raise AssertionError(f"Unsupported mode {mode}")

        if not keep:
            os.remove(str(file))

    def get_plain_archive_name(self, archive_url: str) -> str:
        file_name = re.sub(r".+/", "", archive_url)

        if file_name.endswith(".tar.gz"):
            file_name = file_name.replace(".tar.gz", "")
        elif file_name.endswith(".gz"):
            file_name = file_name.replace(".gz", "")
        elif file_name.endswith(".zip"):
            file_name = file_name.replace(".zip", "")
        elif file_name.endswith(".tar"):
            file_name = file_name.replace(".tar", "")

        return file_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resources", type=str, required=True, nargs="+",
                        choices=["pubtator_central", "disease_ontology"],
                        help="Resource to download")

    # Optional parameters
    parser.add_argument("--resource_dir", type=str, required=False, default="_resources",
                        help="Path to the directory storing the resources")
    args = parser.parse_args()

    archives_to_download = []
    files_to_download = []

    if "pubtator_central" in args.resources:
        archives_to_download += [
            (
                "pubtator_central",
                "ftp://ftp.ncbi.nlm.nih.gov/pub/lu/PubTatorCentral/bioconcepts2pubtatorcentral.offset.gz"
            )
        ]

    if "disease_ontology" in args.resources:
        files_to_download += [
            (
                "do",
                "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/master/src/ontology/doid.obo"
            )
        ]

    resource_dir = Path(args.resource_dir)

    downloader = ResourceDownloader()
    for resource, archive_url in archives_to_download:
        downloader.download_and_unzip_archive(archive_url, resource_dir / resource)

    for resource, file_url in files_to_download:
        downloader.download_file(file_url, resource_dir / resource)

    if "disease_ontology" in args.resources:
        resources = ResourceHandler(resource_dir)

        DiseaseOntologyHandler().prepare_ontology(
            obo_file=resources.get_disease_ontology_obo_file(),
            output_file=resources.get_disease_ontology_tsv_file()
        )




