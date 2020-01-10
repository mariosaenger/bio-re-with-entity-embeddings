import argparse
import os
import pandas as pd

from ast import literal_eval
from collections import defaultdict
from typing import Callable, Set, List
from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from data.disease_ontology import DiseaseOntology
from data.resources import DO_ONTOLOGY_FILE, DO_CANCER_ONTOLOGY_FILE, CIVIC_DATA_SET_FILE, DOCM_DATA_SET_FILE, \
    PHARMA_DATA_SET_FILE
from utils.log_utils import LoggingMixin
from utils.pandas_utils import PandasUtil as pdu, PipelineMixin, EntryFilter, NullOperation


class MutationDiseaseDataPreparation(LoggingMixin):

    def __init__(self, working_directory: str):
        super(MutationDiseaseDataPreparation, self).__init__()
        self.working_directory = working_directory

        self.disease_ontology = DiseaseOntology(DO_ONTOLOGY_FILE)
        self.cancer_ontology = DiseaseOntology(DO_CANCER_ONTOLOGY_FILE)

    def run(self, neg_sample_strategy: str) -> None:
        self.log_info(f"Start mutation-disease preparation with sampling strategy {neg_sample_strategy}")

        # Load co-occurrences from pubtator
        pub_co_occurrences = self.load_cooccurrenes_from_pubtator()

        # Load prepared gold standard data set + merge them
        # -----------------------------------------------------------------------------------------------------------
        civic_df = self.read_prepared_data_set("civic", CIVIC_DATA_SET_FILE)
        docm_df = self.read_prepared_data_set("docm", DOCM_DATA_SET_FILE)
        pharma_df = self.read_prepared_data_set("pharma_gkb", PHARMA_DATA_SET_FILE)

        mutation_disease_df = self.merge_data_sets([civic_df, docm_df, pharma_df])

        gold_dir = os.path.join(self.working_directory, "goldstandard")
        os.makedirs(gold_dir, exist_ok=True)

        gold_file = os.path.join(gold_dir, "goldstandard_pairs.tsv")
        mutation_disease_df.to_csv(gold_file, sep="\t", encoding="utf8")

        pos_sample_file = os.path.join(self.working_directory, "pos_samples.tsv")
        pos_pubmed_file = os.path.join(self.working_directory, "pos_pubmed_ids.txt")
        articles_compare_file = os.path.join(self.working_directory, "article_compare.txt")

        pos_pipeline = Pipeline([
            # Append articles references based on PubTator to the data set
            ("AddArticlesFromPubtator", AddPubtatorCooccurrences(pub_co_occurrences)),

            # Consider only instances where Pubtator has found at least one co-occurrence
            ("FilterInstancesWithText", pdu.not_null("pubtator_articles")),

            # Count number of articles found by Pubtator and given in the original data sets
            ("CountPubtatorArticleclass", pdu.count_values("pubtator_articles", "num_pubtator_articles")),
            ("CountCivicArticlesPerPair", pdu.count_values("articles", "num_articles")),
            ("LimitNumArticles", pdu.map("num_pubtator_articles", self.bucketize_article_count(), "num_articles_norm")),

            ("ExtractPubMedIds", pdu.extract_unique_values("pubtator_articles", pos_pubmed_file)),

            ("ConvertArticlesToString", pdu.map("pubtator_articles", self.set_to_string, "articles_str")),

            ("SaveSamplesToCsv", pdu.to_csv(pos_sample_file, columns=["source_id", "target_id", "source", "articles_str"])),
            ("SaveArticleCompare", pdu.to_csv(articles_compare_file, columns=["source_id", "target_id",
                                               "num_articles", "num_pubtator_articles", "articles_overlap"]))
        ])

        pos_instances = pos_pipeline.fit_transform(mutation_disease_df)
        self.log_info("Finished positive pipeline. Positive set contains %s instances", len(pos_instances))

        # Prepare negative samples
        # -----------------------------------------------------------------------------------------------------------
        neg_sample_dir = os.path.join(self.working_directory, "neg_samples")
        os.makedirs(neg_sample_dir, exist_ok=True)

        num_article_distribution = pos_instances["num_articles_norm"].value_counts(normalize=True)

        neg_sample_file = os.path.join(self.working_directory, "neg_samples.tsv")
        neg_pubmed_file = os.path.join(self.working_directory, "neg_pubmed_ids.txt")

        if neg_sample_strategy == "sample":
            num_neg_samples = len(pos_instances) * 4

            neg_pipeline = Pipeline([
                # Remove pairs form the goldstandard data set
                ("FilterPositiveInstances", RemovePositiveInstances(pos_instances)),

                # Sample negative instance based on the number of articles
                ("SampleInstances", NegativePairSampler(num_neg_samples, self.disease_ontology, pos_instances)),

                # Calculate article statistic
                ("CountArticlesPerPair", pdu.count_values("articles", "num_articles")),
                ("LimitArticleCount", pdu.map("num_articles", self.bucketize_article_count(), "num_articles_norm")),

                # Sample negative instance based on the number of articles
                ("SampleInstances", PairSampler(num_neg_samples, num_article_distribution)),

                # Save data set
                ("ExtractPubMedIds", pdu.extract_unique_values("articles", neg_pubmed_file)),
                ("SaveInstancesAsTsv", pdu.to_csv(neg_sample_file, columns=["source_id", "target_id", "articles_str"]))
            ])

            neg_sample = neg_pipeline.fit_transform(pub_co_occurrences)

        elif neg_sample_strategy in ["tree_sample", "pos_sample_one", "pos_sample_both"]:
            num_neg_samples = len(pos_instances) * 4

            if neg_sample_strategy == "pos_sample_one":
                entry_filter = EntryFilter(pos_instances, False)
            elif neg_sample_strategy == "pos_sample_both":
                entry_filter = EntryFilter(pos_instances, True)
            else:
                entry_filter = NullOperation()

            neg_pipeline = Pipeline([
                # Remove pairs form the goldstandard data set
                ("FilterPositiveInstances", RemovePositiveInstances(pos_instances)),
                ("EntryFilter", entry_filter),

                # Sample negative instance based on the number of articles
                ("SampleInstances", NegativePairSampler(num_neg_samples, self.disease_ontology, pos_instances)),

                # Calculate article statistic
                ("CountArticlesPerPair", pdu.count_values("articles", "num_articles")),
                ("LimitArticleCount", pdu.map("num_articles", self.bucketize_article_count(), "num_articles_norm")),

                # Save data set
                ("ExtractPubMedIds", pdu.extract_unique_values("articles", neg_pubmed_file)),
                ("SaveInstancesAsTsv", pdu.to_csv(neg_sample_file, columns=["source_id", "target_id", "articles_str"]))
            ])

            neg_sample = neg_pipeline.fit_transform(pub_co_occurrences)

        else:
            raise ValueError(f"Unsupported neg sample strategy {neg_sample_strategy}")

        neg_sample.to_csv(os.path.join(neg_sample_dir, "neg_pairs.csv"), sep="\t", index=False)

        self.log_info("Finished negative pipeline. Found %s instances", len(neg_sample))

    def load_cooccurrenes_from_pubtator(self) -> DataFrame:
        # Create sub folders for the results
        pubtator_dir = os.path.join(self.working_directory, "pubtator")
        co_occurrences_file = os.path.join(pubtator_dir, "pubtator_co_occurrences.csv")

        self.log_info("Reloading mutation-disease co-occurrences from %s", co_occurrences_file)
        pub_co_occurrences = pd.read_csv(co_occurrences_file, sep="\t", index_col=None)

        self.log_info("Rebuilding index")
        id_pairs = ["%s#%s" % (row["source_id"], row["target_id"])
                    for i, row in tqdm(pub_co_occurrences.iterrows(), total=len(pub_co_occurrences))]
        pub_co_occurrences.index = id_pairs

        self.log_info("Rebuilding data structures")
        pub_co_occurrences["articles"] = pub_co_occurrences["articles"].map(literal_eval)

        self.log_info("Found %s co-occurring pairs in total", len(pub_co_occurrences))

        return pub_co_occurrences

    def read_prepared_data_set(self, name: str, tsv_file: str) -> DataFrame:
        self.log_info(f"Reading prepared data set from {tsv_file}")
        data_set = pd.read_csv(tsv_file, sep="\t", encoding="utf8")
        data_set.set_index(["source_id", "target_id"], drop=False, inplace=True)
        data_set["source"] = name

        id_to_articles = {}
        for i, row in data_set.iterrows():
            articles = row["articles"]
            if articles is not None and type(articles) == str:
                articles = set(articles.split(","))
            else:
                articles = set()
            id_to_articles[i] = { "articles" : articles }

        data_set = data_set.drop("articles", axis=1)

        article_column = DataFrame.from_dict(id_to_articles, orient="index")
        data_set = pd.concat([data_set, article_column], axis=1)

        self.log_info(f"Found {len(data_set)} instances in total")
        return data_set

    def merge_data_sets(self, data_sets: List[DataFrame]) -> DataFrame:
        self.log_info(f"Merging {len(data_sets)} data sets")
        merge_dict = {}

        for data_set in data_sets:
            for id, row in data_set.iterrows():
                if id not in merge_dict:
                    merge_dict[id] = {
                        "source_id": row["source_id"],
                        "target_id": row["target_id"],
                        "articles":  row["articles"],
                        "source":    row["source"],
                        "origin_id": row["source"] + "-" + str(row["origin_id"])
                    }
                else:
                    merge_dict[id]["articles"].update(set(row["articles"]))
                    merge_dict[id]["source"] = merge_dict[id]["source"] + "," + row["source"]
                    merge_dict[id]["origin_id"] = merge_dict[id]["origin_id"] + "," + row["source"] + "-" + str(row["origin_id"])

        merged_df = pd.DataFrame.from_dict(merge_dict, orient="index")
        self.log_info(f"Finished merging. Data set contains {len(merged_df)} in total")

        return merged_df

    def bucketize_article_count(self) -> Callable:
        def __bucketize_article_count(value):
            if value == 1:
                return "1"
            elif 2 <= value <= 3:
                return "2-3"
            elif 4 <= value <= 8:
                return "4-8"
            else:
                return ">8"

        return __bucketize_article_count

    @staticmethod
    def set_to_string(values):
        if len(values) == 0:
            return None

        return ";;;".join([str(value) for value in sorted(values)])


class PairSampler(PipelineMixin):

    def __init__(self, sample_size: int, num_article_dist: Series):
        super(PairSampler, self).__init__()

        self.sample_size = sample_size
        self.num_article_dist = num_article_dist

    def transform(self, data: DataFrame, y=None):
        result = None

        for num_articles, rate in self.num_article_dist.iteritems():
            num_samples = int(self.sample_size * rate)
            sample_foundation = data.loc[data["num_articles_norm"] == num_articles]

            if num_samples > len(sample_foundation):
                raise Exception("Can't sample %s sample_foundation from base set with %s sample_foundation" %
                                (num_samples, len(sample_foundation)))

            sample = sample_foundation.sample(num_samples, random_state=777)
            if result is None:
                result = sample
            else:
                result = pd.concat([result, sample])

        if len(result) < self.sample_size:
            not_already_sampled = data.drop(result.index)
            sample = not_already_sampled.sample(self.sample_size - len(result))
            result = pd.concat([result, sample])

        elif len(result) > self.sample_size:
            result = result.sample(self.sample_size, random_state=777)

        return result


class NegativePairSampler(PipelineMixin):

    def __init__(self, sample_size: int, disease_ontology: DiseaseOntology, positive_pairs: DataFrame):
        super(NegativePairSampler, self).__init__()

        self.sample_size = sample_size
        self.disease_ontology = disease_ontology
        self.positive_pairs = positive_pairs

        self.mutation_to_disease = None
        self.disease_to_pos_prefixes = None
        self.disease_to_paths = None

    def transform(self, data, y=None, **fit_params):
        self.log_info(f"Start sampling of {self.sample_size} negative pairs")
        # Initialize auxiliary data structures
        self.log_info("Initialize mappings and auxiliary data structures")
        self._initialize_mappings()

        instances_to_sample = self.sample_size
        result = None

        while instances_to_sample > 0:
            self.log_info(f"Sampling {instances_to_sample} from data set (size: {len(data)})")
            samples = data.sample(instances_to_sample, random_state=23)
            invalid_samples = set()

            # Randomly sample instances from candidate set and check whether they are valid
            # according to the positive instances
            for id, row in tqdm(samples.iterrows(), total=len(samples), desc="sample"):
                mutation_id = row["source_id"]
                disease_id = row["target_id"]
                is_invalid = False

                # All paths of candidate negative sample in DO
                neg_paths = self.disease_ontology.get_paths(disease_id)

                pos_disease_ids = self.mutation_to_disease[mutation_id]

                for pos_disease_id in pos_disease_ids:
                    # All prefixes of diseases for this mutation in positive instances
                    pos_prefixes = self.disease_to_pos_prefixes[pos_disease_id]

                    is_invalid = any([path in pos_prefixes for path in neg_paths])
                    if is_invalid:
                        break

                    pos_disease_paths = self.disease_to_paths[pos_disease_id]
                    is_invalid = any([neg_path.startswith(pos_path) for neg_path in neg_paths for pos_path in pos_disease_paths])
                    if is_invalid:
                        break

                if is_invalid:
                    invalid_samples.add(id)

            self.log_info(f"Dropping {len(invalid_samples)} entries from sampled instances")
            valid_samples = samples.drop(invalid_samples)

            if result is None:
                result = valid_samples
            else:
                result = pd.concat([result, valid_samples])

            instances_to_sample = instances_to_sample - len(valid_samples)

            self.log_info(f"Use {len(samples)} entries from complete data set")
            data = data.drop(samples.index.values)

        return result

    def _initialize_mappings(self):
        self.mutation_to_disease = defaultdict(set)
        self.disease_to_pos_prefixes = defaultdict(dict)
        self.disease_to_paths = defaultdict(set)

        for i, row in tqdm(self.positive_pairs.iterrows(), total=len(self.positive_pairs), desc="init"):
            mutation_id = row["source_id"]
            disease_id = row["target_id"]

            self.mutation_to_disease[mutation_id].add(disease_id)

            prefixes = self.disease_ontology.get_path_prefixes_by_doid(disease_id)
            prefixes = { pre : True for pre in prefixes}
            self.disease_to_pos_prefixes[disease_id].update(prefixes)

            paths = self.disease_ontology.get_paths(disease_id)
            self.disease_to_paths[disease_id].update(paths)


class MeshTermToDoidMapper(PipelineMixin):

    def __init__(self, disease_ontology: DiseaseOntology, mesh_extractor: Callable, source_id_extractor: Callable):
        super(MeshTermToDoidMapper, self).__init__()
        self.disease_ontology = disease_ontology
        self.mesh_extractor = mesh_extractor
        self.source_id_extractor = source_id_extractor

    def transform(self, data: DataFrame, y=None):
        self.log_info("Adding doid id to %s instances", len(data))
        num_unknown_doid = 0

        new_data_map = dict()
        for id, row in tqdm(data.iterrows(), total=len(data)):
            disease_mesh_term = self.mesh_extractor(id, row)

            doids = self.disease_ontology.get_doid_by_mesh(disease_mesh_term)
            if len(doids)  > 0:
                source_id = self.source_id_extractor(id, row)
                for doid in doids:
                    row_copy = row.copy()
                    row_copy["doid"] = doid

                    new_row_id = source_id + "#" + doid
                    row_copy["id_doid"] = new_row_id

                    if new_row_id in new_data_map:
                        a1 = row_copy["articles"]
                        a2 = row_copy["articles"]
                        row_copy["articles"] = a1.union(a2)

                    new_data_map[new_row_id] = row_copy
            else:
                num_unknown_doid = num_unknown_doid + 1

        new_data = DataFrame(list(new_data_map.values()))
        new_data.index = new_data["id_doid"]
        new_data = new_data.drop("id_doid", axis=1)

        self.log_info("Can't find DOID for %s of %s entries", num_unknown_doid, len(data))
        self.log_info("Finished mesh to doid mapping. New data set has %s instances", len(new_data))

        return new_data


class AddPubtatorCooccurrences(PipelineMixin):

    def __init__(self, pub_cooccurrences: DataFrame):
        super(AddPubtatorCooccurrences, self).__init__()
        self.pub_cooccurrences = pub_cooccurrences

    def transform(self, data: DataFrame):
        self.log_info("Extending civic data with collected co-occurrences from pubtator")

        articles = []
        overlap = []
        for i, row in tqdm(data.iterrows(), total=len(data)):
            mutation_id = row["source_id"]
            doid = row["target_id"]

            pubtator_articles = set()
            result = self.pub_cooccurrences[(self.pub_cooccurrences["source_id"] == mutation_id) &
                                                (self.pub_cooccurrences["target_id"] == doid)]
            if len(result) == 1:
                for _, line in result.iterrows():
                    pubtator_articles.update(line["articles"])
            elif len(result) > 1:
                self.log_info("Found multiple results for pair: %s-%s", mutation_id, doid)

            #self.log_debug("Found %s articles for pair %s-%s", len(pubtator_articles), mutation_id, doid)
            civic_articles = row["articles"]
            if civic_articles is None or type(civic_articles) == float:
                civic_articles = set()
            elif type(civic_articles) == str:
                self.log_info(civic_articles)

            # Count overlap between pubtator and civic articles
            overlap.append(len(civic_articles.intersection(pubtator_articles)))

            if len(pubtator_articles) > 0:
                articles.append(pubtator_articles)
            else:
                articles.append(None)

        data["articles_overlap"] = overlap
        data["pubtator_articles"] = articles

        return data


class RemovePositiveInstances(PipelineMixin):

    def __init__(self, pos_instances: DataFrame):
        super(RemovePositiveInstances, self).__init__()
        self.pos_instances = pos_instances

    def transform(self, data: DataFrame):
        self.log_info("Removing positive instances from co-occurrence list")
        indexes_to_remove = set()

        for i, row in tqdm(self.pos_instances.iterrows(), total=len(self.pos_instances)):
            row_id = row["source_id"] + "#" + row["target_id"]

            if row_id in data.index:
                indexes_to_remove.add(row_id)

        prev_num_instances = len(data)
        data = data.drop(indexes_to_remove, axis=0)
        self.log_info("Removed %s instances from co-occurrence mapping", prev_num_instances - len(data))

        return data


class AddPositiveInstances(PipelineMixin):

    def __init__(self, positive_instances: DataFrame):
        super(AddPositiveInstances, self).__init__()
        self.positive_instances = positive_instances

    def transform(self, data: DataFrame, y=None):
        self.log_info("Adding positive instances to all pubtator co-occurrences")
        data = data[["source_id", "target_id", "articles"]]

        new_instances = dict()
        for i, row in self.positive_instances.iterrows():
            if i not in data.index:
                new_instances[i] = {
                    "source_id": row["source_id"],
                    "target_id" : row["target_id"],
                    "articles" : row["pubtator_articles"]
                }

        self.log_info("Found %s instances not contained in pubtator co-occurrences", len(new_instances))
        new_data = DataFrame.from_dict(new_instances, orient="index")

        return pd.concat([data, new_data])


class CancerDiseaseFilter(PipelineMixin):

    def __init__(self, cancer_ids: Set, doid_column="doid"):
        super(CancerDiseaseFilter, self).__init__()
        self.cancer_ids = cancer_ids
        self.doid_column = doid_column

    def transform(self, data: DataFrame, y=None):
        self.log_info("Scanning data set for cancer related pairs (%s cancer ids)", len(self.cancer_ids))
        #self.log_info("Cancer ids: \n%s", "\n".join(["'" + id + "'" for id in self.cancer_ids]))

        rows_to_remove = []
        other_ids = set()
        for i, row in tqdm(data.iterrows(), total=len(data)):
            if row[self.doid_column] not in self.cancer_ids:
                rows_to_remove.append(i)
                other_ids.add(row[self.doid_column])

        data = data.drop(rows_to_remove)
        self.log_info("Removed %s instances (%s cancer instances remain)", len(rows_to_remove), len(data))
        return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Script to prepare the mutation disease data")
    parser.add_argument("output_dir", help="Path to the output directory", type=str)
    parser.add_argument("neg_sample_strategy", choices=["sample", "tree_sample", "pos_sample_one", "pos_sample_both"],
                        type=str, help="Indicates which strategy to use for negative samples")

    args = parser.parse_args()

    mutation_disease_data = MutationDiseaseDataPreparation(args.output_dir)
    mutation_disease_data.run(args.neg_sample_strategy)
