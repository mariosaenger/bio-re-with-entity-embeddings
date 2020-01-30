import os
import math
import pandas as pd

from argparse import ArgumentParser
from ast import literal_eval

from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from typing import Callable

from utils.pandas_utils import PipelineMixin
from utils.log_utils import LoggingMixin
from utils.pandas_utils import PandasUtil as pdu


class DrugDrugPreparation(LoggingMixin):

    def __init__(self):
        super(DrugDrugPreparation, self).__init__()

    def run(self, output_dir: str):
        self.log_info(f"Start building drug-drug data set")
        os.makedirs(output_dir, exist_ok=True)

        statistics_dir = os.path.join(output_dir, "stat")
        os.makedirs(statistics_dir, exist_ok=True)

        geneview_dir = os.path.join(output_dir, "geneview")
        os.makedirs(geneview_dir, exist_ok=True)

        self.log_info("Start preparation of drug-drug data set")

        geneview_file = os.path.join(geneview_dir, "geneview.csv")
        self.log_info("Reloading gene view interactions from %s", geneview_file)

        drugbank_data_set = pd.read_csv(geneview_file, sep="\t", index_col="id")
        drugbank_data_set["interactions"] = drugbank_data_set["interactions"].map(self.string_to_set)
        drugbank_data_set["articles"] = drugbank_data_set["articles"].map(self.string_to_set)

        # Prepare positive pairs
        # -----------------------------------------------------------------------------------------
        drugbank_dir = os.path.join(output_dir, "drugbank")
        os.makedirs(drugbank_dir, exist_ok=True)

        positive_pairs_file = os.path.join(drugbank_dir, "positive_pairs.tsv")
        if not os.path.exists(positive_pairs_file):
            self.log_info("Start building positive drug interaction pairs")
            positive_pairs = self.build_positive_pairs(drugbank_data_set)
            positive_pairs.to_csv(positive_pairs_file, sep="\t", index_label="id")
            self.log_info("Found %s positive drug interaction pairs", len(positive_pairs))
        else:
            self.log_info("Reloading positive pairs from %s", positive_pairs_file)
            positive_pairs = pd.read_csv(positive_pairs_file, sep="\t", index_col="id")
            positive_pairs["articles"] = positive_pairs["articles"].map(self.string_to_set)
            self.log_info("Found %s positive drug interaction pairs", len(positive_pairs))

        pos_sample_file = os.path.join(output_dir, "pos_samples.tsv")
        pos_pubmed_file = os.path.join(output_dir, "pos_pubmed_ids.txt")

        positive_ds_pipeline = Pipeline([
            # Only consider pairs which occur at least once!
            ("FilterPairsWithoutArticles", pdu.not_null("articles")),

            ("CountArticlesPerPair", pdu.count_values("articles", "num_articles")),
            ("LimitArticleCount", pdu.map("num_articles", self.min_article_count(30), "num_articles_norm")),

            ("ConvertArticlesToString", pdu.map("articles", self.set_to_string, "articles_str")),

            ("SaveArticleIds", pdu.extract_unique_values("articles", pos_pubmed_file)),
            ("SaveDrugDataSet", pdu.to_csv(pos_sample_file, columns=["source_id", "target_id", "articles_str"]))
        ])

        pos_data_set = positive_ds_pipeline.fit_transform(positive_pairs)
        self.log_info("Found %s positive drug interactions with at least one article", len(pos_data_set))

        # Prepare negative pairs
        # -----------------------------------------------------------------------------------------
        cooccurring_pairs_file = os.path.join(geneview_dir, "cooccurring_pairs.pd")
        cooccurring_instances_file = os.path.join(geneview_dir, "pair_instances.tsv")
        cooccurring_pubmed_ids_file = os.path.join(geneview_dir, "drug_pubmed_ids.txt")

        if not os.path.exists(cooccurring_pairs_file):
            self.log_info("Building co-occurrence pairs")
            cooccurring_pairs = self.get_cooccurring_drug_pairs(drugbank_data_set)
            cooccurring_pairs["articles"] = cooccurring_pairs["articles"].map(self.empty_set_to_none)
            cooccurring_pairs.to_csv(cooccurring_pairs_file, sep="\t", index_label="id", header=True)
            self.log_info("Found %s co-occurring drug pairs", len(cooccurring_pairs))

            cooccurrence_pipeline = Pipeline([
                ("ConvertArticlesToString", pdu.map("articles", self.set_to_string, "articles_str")),
                ("ExtractPubMedIds", pdu.extract_unique_values("articles", cooccurring_pubmed_ids_file)),
                ("SaveInstancesAsTsv", pdu.to_csv(cooccurring_instances_file, columns=["source_id", "target_id", "articles_str"]))
            ])

            cooccurrence_pipeline.fit_transform(cooccurring_pairs)
            self.log_info("Saved co-occurring instances to tsv")

        else:
            self.log_info("Reloading co-occurring pairs from %s", cooccurring_pairs_file)
            cooccurring_pairs = pd.read_csv(cooccurring_pairs_file, sep="\t", index_col="id")
            cooccurring_pairs["articles"] = cooccurring_pairs["articles"].map(self.string_to_set)
            self.log_info("Found %s co-occurring drug pairs", len(cooccurring_pairs))

        self.log_info("Removing positive pairs from co-occurrence data set")
        cooccurring_pairs = self.remove_positive_pairs(cooccurring_pairs, positive_pairs)

        num_article_distribution = pos_data_set["num_articles_norm"].value_counts(normalize=True)

        neg_sample_file = os.path.join(output_dir, "neg_samples.tsv")
        neg_pubmed_file = os.path.join(output_dir, "neg_pubmed_ids.txt")

        num_neg_samples = len(pos_data_set) * 4

        negative_ds_pipeline = Pipeline([
            ("CountArticlesPerPair", pdu.count_values("articles", "num_articles")),
            ("LimitArticleCount", pdu.map("num_articles", self.min_article_count(30), "num_articles_norm")),

            ("SampleInstances", PairSampler(num_neg_samples, num_article_distribution)),

            ("ExtractPubMedIds", pdu.extract_unique_values("articles", neg_pubmed_file)),

            ("RenameColumnNames", pdu.rename_columns({"id1": "source_id", "doid": "target_id"})),
            ("ConvertArticlesToString", pdu.map("articles", self.set_to_string, "articles_str")),
            ("SaveInstancesAsTsv", pdu.to_csv(neg_sample_file, columns=["source_id", "target_id", "articles_str"]))
        ])

        self.log_info(f"Starting negative pipeline to sample {num_neg_samples} instances")
        negative_ds_pipeline.fit_transform(cooccurring_pairs)
        self.log_info("Finished data preparation")

    def build_positive_pairs(self, drugbank_data_set: DataFrame) -> DataFrame:
        positive_pairs = dict()

        for drug_id, row in tqdm(drugbank_data_set.iterrows(), total=len(drugbank_data_set)):
            articles = row["articles"]
            interactions = row["interactions"]

            if interactions is None:
                continue

            for other_drug_id in interactions:
                pair_id = self.build_pair_id(drug_id, other_drug_id)
                if pair_id in positive_pairs:
                    continue

                source_id = pair_id.split("#")[0]
                target_id = pair_id.split("#")[1]

                intersection = None

                if other_drug_id in drugbank_data_set.index:
                    other_articles = drugbank_data_set.loc[other_drug_id]["articles"]
                    if articles is not None and other_articles is not None:
                        intersection = articles.intersection(other_articles)

                    if intersection is None or len(intersection) == 0:
                        intersection = None

                positive_pairs[pair_id] = { "source_id": source_id, "target_id": target_id, "articles" : intersection}

        positive_pairs_df = pd.DataFrame.from_dict(positive_pairs, orient="index")
        return positive_pairs_df

    def get_cooccurring_drug_pairs(self, drugbank_data_set: DataFrame) -> DataFrame:
        self.log_debug("Building pubmed id to drugs mapping")
        article_to_drugs_map = dict()
        for id, row in tqdm(drugbank_data_set.iterrows(), total=len(drugbank_data_set)):
            articles = row["articles"]
            if articles is None:
                continue

            for pubmed_id in articles:
                if pubmed_id not in article_to_drugs_map:
                    article_to_drugs_map[pubmed_id] = {"pubmed_id": pubmed_id, "drug_ids": set()}

                article_to_drugs_map[pubmed_id]["drug_ids"].add(id)
        self.log_debug("Found %s distinct articles with at least one drug", len(article_to_drugs_map))

        self.log_debug("Building co-occurring drug pairs")
        pair_to_articles_map = dict()
        for pubmed_id, value in tqdm(article_to_drugs_map.items(), total=len(pair_to_articles_map)):
            drug_ids = value["drug_ids"]
            pair_ids = [self.build_pair_id(drug_id1, drug_id2) for drug_id1 in drug_ids
                        for drug_id2 in drug_ids if drug_id1 != drug_id2]

            for pair_id in pair_ids:
                source_id = pair_id.split("#")[0]
                target_id = pair_id.split("#")[1]

                if pair_id not in pair_to_articles_map:
                    pair_to_articles_map[pair_id] = { "pair_id" : pair_id, "source_id": source_id,
                                                      "target_id": target_id, "articles": set()}

                pair_to_articles_map[pair_id]["articles"].add(pubmed_id)

        return pd.DataFrame.from_dict(pair_to_articles_map, orient="index")

    def remove_positive_pairs(self, cooccurring_pairs: DataFrame, positive_pairs: DataFrame) -> DataFrame:
        original_size = len(cooccurring_pairs)
        ids_to_remove = positive_pairs.loc[positive_pairs["articles"].notnull()].index
        reduced_cooccuring_pairs = cooccurring_pairs.drop(ids_to_remove)
        new_size = len(reduced_cooccuring_pairs)
        self.log_info("Removed %s positive pairs from co-occurrences (new size: %s)", original_size - new_size, new_size)
        return reduced_cooccuring_pairs

    @staticmethod
    def min_article_count(threshold: int) -> Callable:
        def __min_article_count(value):
            return min(int(value), threshold)
        return __min_article_count

    @staticmethod
    def build_pair_id(drug1: str, drug2: str) -> str:
        if drug1 < drug2:
            return drug1 + "#" + drug2
        elif drug1 > drug2:
            return drug2 + "#" + drug1
        else:
            raise Exception("A drug can't interact with itself!")

    @staticmethod
    def string_to_set(value):
        if value is None:
            return None

        if isinstance(value, float) and math.isnan(value):
            return None

        return literal_eval(str(value))

    @staticmethod
    def set_to_string(values):
        if values is None or len(values) == 0:
            return None

        return ";;;".join([str(value) for value in values])

    @staticmethod
    def empty_set_to_none(value):
        if len(value) == 0:
            return None
        return value


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

            sample = sample_foundation.sample(num_samples, random_state=42)
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("output_dir", help="Path to the output directory", type=str)
    args = parser.parse_args()

    drug_drug_preparation = DrugDrugPreparation()
    drug_drug_preparation.run(args.output_dir)



