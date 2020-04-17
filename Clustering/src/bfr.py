import csv
import json
import math
import os
import sys
import time

from pyspark import SparkContext
from pyspark.rdd import RDD

SparkContext.setSystemProperty('spark.executor.memory', '7g')
SparkContext.setSystemProperty('spark.driver.memory', '7g')
sc = SparkContext(appName="INF553HW5", master="local[*]")
sc.setLogLevel("WARN")
sc.setLogLevel("ERROR")


def main(argv):
    input_path = argv[1]
    num_clusters = int(argv[2])
    clustering_file_path = argv[3]
    intermediate_file_path = argv[4]

    # intermediate_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW5/output/intermediate-result.csv"
    # clustering_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW5/output/clustering-result-test5.json"
    # input_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW5/data/test5/"
    # num_clusters = 15

    def standardize(data: RDD) -> (RDD, set):
        """
        Standardize the input data_.
        :param data: pair RDD, key is index of the point, value is the features of the point
        :return: the standardized data_ and a set of ids of outlier points
        """
        def compute_mean_std(data_) -> tuple:
            intermediate = data_. \
                values(). \
                flatMap(lambda features: [(i, feature) for i, feature in enumerate(features)]). \
                persist()

            count = data_.count()
            mean = intermediate.reduceByKey(lambda x, y: x + y).mapValues(lambda value: value / count).persist()
            mean_dict = dict(mean.collect())

            std = intermediate.\
                mapValues(lambda value: value ** 2).\
                reduceByKey(lambda x, y: x + y). \
                mapValues(lambda value: value / count). \
                map(lambda pair: (pair[0], pair[1] - mean_dict[pair[0]] ** 2)). \
                mapValues(lambda values: math.sqrt(values)). \
                persist()
            std_dict = dict(std.collect())
            return mean_dict, std_dict

        mean_dict, std_dict = compute_mean_std(data)
        outliers_ = data. \
            mapValues(lambda values: [abs((value - mean_dict[i]) / std_dict[i]) for i, value in enumerate(values)]). \
            mapValues(max). \
            filter(lambda pair: pair[1] > 3). \
            keys().collect()
        outliers_ = set(outliers_)
        data = data.filter(lambda pair: pair[0] not in outliers_).persist()

        mean_dict, std_dict = compute_mean_std(data)
        standardized_data = data. \
            mapValues(lambda values: [(value - mean_dict[index]) / std_dict[index] for index, value in enumerate(values)]). \
            persist()
        return standardized_data, outliers_

    def kmeans(data: RDD, num_clusters: int, min_cluster_id=0, max_iterations=20) -> RDD:
        """
        :param data: pair RDD, key is point_id, value is feature vector
        :param num_clusters: number of clusters
        :param min_cluster_id: The id of each cluster is min_cluster_id, min_cluster_id + 1, min_cluster_id + 2 ...
        :param max_iterations: maximum number of iterations
        :return: clustering result, pair RDD, key is point_id, value is (cluster id, feature vector)
        """

        # ** some helper functions:
        def minimum_distance_to_centroids(point: list, centroids: dict) -> float:
            min_distance = min([sum([(i - j) ** 2 for i, j in zip(point, centroid)]) for centroid in centroids])
            return min_distance

        def assign_point_to_cluster(point: list, centroids_: dict) -> int:
            "Given a data point, find the nearest centroid to it"
            distances = [(sum([(i - j) ** 2 for i, j in zip(point, centroid)]), cid)
                          for cid, centroid in centroids_.items()]
            min_distance, cluster_index = min(distances)
            return cluster_index

        # ** initialize the first k centroids
        total_samples = data.count()
        print("initializing centroids")
        print("total samples:", total_samples)
        sample_fraction = 0.3
        if total_samples * sample_fraction < num_clusters:
            if total_samples < num_clusters:
                centroids = data.values().collect()
            else:
                centroids = data.values().take(num_clusters)
        else:
            sample_data = data.sample(withReplacement=False, fraction=sample_fraction, seed=10)
            centroids = [sample_data.first()[1]]  # add first centroid to the collection of centroids
            for _ in range(num_clusters - 1):  # find the next k-1 centroids
                distances = sample_data.\
                    mapValues(lambda values: minimum_distance_to_centroids(values, centroids)).persist()
                furthest_point = sample_data.lookup(distances.max(lambda pair: pair[1])[0])[0]
                centroids.append(furthest_point)   # furthest from already selected points

        centroids = [(min_cluster_id + i, centroid) for i, centroid in enumerate(centroids)]  # assign index
        centroids = dict(centroids)

        # ** perform clustering
        num_iterations = 0
        previous_cluster_sizes = dict()
        while num_iterations < max_iterations:
            print("current iteration:", num_iterations)
            print("assign points to clusters")
            cluster_result: RDD = data.mapValues(
                lambda values: (assign_point_to_cluster(values, centroids), values)).persist()

            cluster_sizes = cluster_result.map(lambda pair: (pair[1][0], 1)).reduceByKey(lambda x, y: x + y)
            cluster_sizes = dict(cluster_sizes.collect())
            if num_iterations > 0:  # after first iteration
                num_reassigments = [(cluster_sizes[cid] - previous_cluster_sizes[cid]) for cid in cluster_sizes.keys()]
                print("num_reassigments:", num_reassigments)
                if max(num_reassigments) - min(num_reassigments) < 3:  # Now the process converges
                    break

            previous_cluster_sizes = cluster_sizes.copy()
            print("update centroids_:")
            new_centroids = cluster_result. \
                values(). \
                flatMapValues(lambda features: [(i, feature) for i, feature in enumerate(features)]). \
                map(lambda pair: ((pair[0], pair[1][0]), (pair[1][1], 1))). \
                reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])). \
                mapValues(lambda values: values[0] / values[1]). \
                map(lambda pair: (pair[0][0], (pair[0][1], pair[1]))). \
                groupByKey(). \
                mapValues(sorted). \
                mapValues(lambda values: [value for feature_index, value in values]). \
                persist()
            new_centroids = dict(new_centroids.collect())
            centroids = new_centroids.copy()
            num_iterations += 1

        print("Converged. Total iterations:", num_iterations)
        print("cluster size", cluster_result.values().mapValues(lambda values: 1).groupByKey().mapValues(len).collect())
        return cluster_result

    def bfr(input_file_list_: list, num_clusters_: int, alpha: int = 10, ) -> tuple:
        """
        Perform BFR algorithm on all chunks of data the in the input list
        :param input_file_list_: chunks of data to be processed
        :param num_clusters_: Number of clusters in the discard set
        :param alpha: Hyper-parameter controlling
        :return: summary information of discard set, id of outlier points, intermediate results during processing
        """

        def summary_cluster(cluster_result_: RDD) -> dict:
            """
            Get the summary information of the discard set/ compression set
            :param cluster_result_: pair RDD, key is point_id, value is (cluster_id, feature vector)
            :return: a dictionary containing the summary information
            """
            cluster_sum = cluster_result_. \
                values(). \
                flatMapValues(lambda features: [(feature_index, (feature, feature ** 2))
                                                for feature_index, feature in enumerate(features)]). \
                map(lambda pair: ((pair[0], pair[1][0]), pair[1][1])). \
                reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])). \
                map(lambda pair: (pair[0][0], (pair[0][1], (pair[1][0], pair[1][1])))). \
                groupByKey(). \
                mapValues(sorted). \
                mapValues(lambda values: [value for feature_index, value in values]). \
                mapValues(lambda values: [value for value, _ in values]).\
                persist()
            cluster_sum = dict(cluster_sum.collect())

            cluster_assignments = dict(
                cluster_result_.map(lambda pair: (pair[1][0], pair[0])).groupByKey().mapValues(set).collect())
            cluster_point_index = set(cluster_result_.keys().collect())
            cluster_count = dict(
                cluster_result_.map(lambda pair: (pair[1][0], 1)).reduceByKey(lambda x, y: x + y).collect())
            cluster_centroids = dict()
            for cid in cluster_sum.keys():
                cluster_size = cluster_count[cid]
                cluster_centroids[cid] = [Sum / cluster_size for Sum in cluster_sum[cid]]

            return {"assignments": cluster_assignments,
                    "point_index": cluster_point_index,
                    "count": cluster_count,
                    "sum": cluster_sum,
                    "centroids": cluster_centroids}

        def assign_point_to_set(point: list, cluster_summary: dict) -> int:
            """
            Compute the shortest distance between a point and centroids of discard set or compressed set,
            if it's shorter than the threshold, assign this point to the corresponding cluster, otherwise return -1
            :param point: a feature vector
            :param cluster_summary: summary information of discard set or compressed set
            :return: cluster id of the nearest cluster or -1
            """
            cluster_centroids = cluster_summary["centroids"]
            distances = [(math.sqrt(sum([(i - j) ** 2 for i, j in zip(point, centroid)])), cluster_id)
                         for cluster_id, centroid in cluster_centroids.items()]
            min_distance, cluster_id = min(distances)
            if min_distance < alpha * math.sqrt(num_features):
                return cluster_id
            else:
                return -1

        def update_summary(summary: dict, new_data: RDD) -> dict:
            """
            Update the summary information with the newly added data points
            :param summary: summary information of discard set or compressed set
            :param new_data: Newly added data points that are close enough to discard set or compressed set
            :return: The updated summary information of discard set or compressed set
            """
            add_summary = summary_cluster(new_data)
            summary["point_index"] = summary["point_index"].union(add_summary["point_index"])
            for cid in add_summary["assignments"].keys():
                if cid in summary["assignments"]:
                    summary["assignments"][cid] = summary["assignments"][cid].union(add_summary["assignments"][cid])
                    summary["count"][cid] += add_summary["count"][cid]
                    summary["sum"][cid] += add_summary["sum"][cid]
                    count = summary["count"][cid]
                    summary["centroids"][cid] = [Sum / count for Sum in summary["sum"][cid]]
                else:
                    summary["assignments"][cid] = add_summary["assignments"][cid]
                    summary["count"][cid] = add_summary["count"][cid]
                    summary["sum"][cid] = add_summary["sum"][cid]
                    summary["centroids"][cid] = add_summary["centroids"][cid]
            return summary

        def combine_compressed_set(compressed_summary_: dict, new_summary_: dict) -> dict:
            """
            If distance is lower than threshold, map new cluster id to previous cluster id
            :param compressed_summary_: summary information of compressed set
            :param new_summary_: summary information of newly added compressed points
            :return: A mapping that maps cluster id in new points to a new id
            """
            centroids = compressed_summary_["centroids"]
            new_centroids = new_summary_['centroids']
            mapping = {}
            num_previous_clusters = len(centroids)

            for cid, new_centroid in new_centroids.items():
                distances = []
                for i in centroids.keys():
                    centroid = centroids[i]
                    distances.append((math.sqrt(sum([(i - j) ** 2 for i, j in zip(new_centroid, centroid)])), i))

                min_distance, old_cid = min(distances)
                if min_distance < alpha * math.sqrt(num_features):
                    mapping[cid] = old_cid  # assign the new cluster to previous cluster
                else:
                    mapping[cid] = cid

            num_not_combined_clusters = 0
            for before_id, after_id in mapping.items():
                if before_id == after_id:  # means this cluster doesn't merge to previous clusters in CS
                    mapping[before_id] = num_previous_clusters + num_not_combined_clusters
                    num_not_combined_clusters += 1
            return mapping

        def combine_discard_compressed(discard_summary_: dict, compressed_summary_: dict) -> set:
            """
            Combine the summary of discard set and compressed set, and return the outliers_ that still remain in cs
            :param discard_summary_: summary information of discard set
            :param compressed_summary_: summary information of compressed set
            :return: a set of point ids of the outliers_ in compressed set
            """
            ds_centroids = discard_summary_["centroids"]
            cs_centroids = compressed_summary_["centroids"]
            outliers = set()
            for cs_id, cs_centroid in cs_centroids.items():
                distances = []
                for i in ds_centroids.keys():
                    ds_centroid = ds_centroids[i]
                    distances.append((math.sqrt(sum([(i - j) ** 2 for i, j in zip(cs_centroid, ds_centroid)])), i))

                min_distance, old_cid = min(distances)
                if min_distance < alpha * math.sqrt(num_features):
                    discard_summary_["assignments"][old_cid] = discard_summary_["assignments"][old_cid]. \
                        union(compressed_summary_["assignments"][cs_id])
                else:
                    outliers = outliers.union(compressed_summary_["assignments"][cs_id])

            for point_index in discard_summary_["assignments"].values():
                discard_summary_["point_index"] = discard_summary_["point_index"].union(point_index)

            return outliers

        many_clusters = num_clusters_ * 3
        outliers_ = set()

        num_clusters_ds = []
        num_points_ds = []
        num_clusters_cs = []
        num_points_cs = []
        num_points_rs = []

        for number_of_file in range(len(input_file_list_)):
            file_path = input_file_list_[number_of_file]
            print("N.O chunk: ", number_of_file, file_path)
            input_file_path = input_path + file_path

            start_time_for_this_iteration = time.time()
            data = sc.textFile(input_file_path). \
                map(lambda line: list(map(float, line.split(",")))). \
                map(lambda line: (line[0], line[1:])). \
                persist()
            data, outliers_in_chunk = standardize(data)
            outliers_ = outliers_.union(outliers_in_chunk)
            print("Outliers in this chunk:", len(outliers_in_chunk), "Total outliers_:", len(outliers_))
            num_features = len(data.first()[1])

            if number_of_file == 0:  # ** first round
                sample_data = data.sample(withReplacement=False, fraction=0.8, seed=666)
                sample_cluster_result = kmeans(sample_data, num_clusters=num_clusters_)
                discard_summary_ = summary_cluster(sample_cluster_result)

                retained_data = data.filter(lambda pair: pair[0] not in discard_summary_["point_index"]).persist()
                cluster_result = kmeans(retained_data, num_clusters=many_clusters)

                singleton_clusters = cluster_result. \
                    map(lambda pair: (pair[1][0], pair[0])). \
                    groupByKey(). \
                    mapValues(len). \
                    filter(lambda pair: pair[1] == 1). \
                    collect()
                singleton_clusters = set(dict(singleton_clusters).keys())
                retained_data = cluster_result. \
                    filter(lambda pair: pair[1][0] in singleton_clusters). \
                    mapValues(lambda values: values[1]). \
                    persist()
                compressed_data = cluster_result.filter(lambda pair: pair[1][0] not in singleton_clusters).persist()
                compressed_summary = summary_cluster(compressed_data)

            else:  # other rounds
                # finish updating discard set
                print("Assign new points based on Mahalanobis distance:")
                new_discard_points = data. \
                    mapValues(lambda values: (assign_point_to_set(values, discard_summary_),
                                              values)). \
                    filter(lambda pair: pair[1][0] != -1). \
                    persist()
                print("new_discard_points:", new_discard_points.count())
                discard_summary_ = update_summary(discard_summary_, new_discard_points)

                new_compressed_points = data. \
                    filter(lambda pair: pair[0] not in discard_summary_["point_index"]). \
                    mapValues(lambda values: (assign_point_to_set(values, compressed_summary),
                                              values)). \
                    filter(lambda pair: pair[1][0] != -1). \
                    persist()
                print("new_compressed_points:", new_compressed_points.count())
                compressed_summary = update_summary(compressed_summary, new_compressed_points)

                retained_data = data. \
                    filter(lambda pair: (pair[0] not in discard_summary_["point_index"]) and
                                        (pair[0] not in compressed_summary["point_index"])). \
                    union(retained_data). \
                    persist()

                # finish updating retained set
                if not retained_data.isEmpty():
                    print("perform K-Means on retained set:")
                    cluster_result = kmeans(retained_data,
                                            min_cluster_id=len(compressed_summary["count"]),
                                            num_clusters=many_clusters)

                    singleton_clusters = cluster_result. \
                        map(lambda pair: (pair[1][0], pair[0])). \
                        groupByKey(). \
                        mapValues(len). \
                        filter(lambda pair: pair[1] == 1). \
                        collect()
                    singleton_clusters = set(dict(singleton_clusters).keys())
                    retained_data = cluster_result. \
                        filter(lambda pair: pair[1][0] in singleton_clusters). \
                        mapValues(lambda values: values[1]). \
                        persist()

                    # finish updating compressed set
                    new_compressed_data = cluster_result.filter(
                        lambda pair: pair[1][0] not in singleton_clusters).persist()
                    if not new_compressed_data.isEmpty():
                        print("Merge clusters in compressed set:")
                        new_summary = summary_cluster(new_compressed_data)

                        mapping = combine_compressed_set(compressed_summary, new_summary)
                        new_compressed_data = new_compressed_data.mapValues(
                            lambda values: (mapping[values[0]], values[1])).persist()
                        compressed_summary = update_summary(compressed_summary, new_compressed_data)
                else:
                    print("empty retained set, process next chunk")

            # ** compute intermediate results in each iteration
            num_clusters_ds.append(len(discard_summary_["assignments"]))
            num_points_ds.append(len(discard_summary_["point_index"]))
            num_clusters_cs.append(len(compressed_summary["assignments"]))
            num_points_cs.append(len(compressed_summary["point_index"]))
            num_points_rs.append(retained_data.count())
            print("intermediate results in each iteration:",
                  num_clusters_ds, num_points_ds, num_clusters_cs, num_points_cs, num_points_rs)
            print("cluster_size", sorted(list(discard_summary_["count"].values())))
            print("running time for this iteration:", time.time() - start_time_for_this_iteration, "\n")

        outliers_from_cs = combine_discard_compressed(discard_summary_, compressed_summary)
        outliers_from_rs = set(retained_data.keys().collect())
        outliers_from_bfr = outliers_from_cs.union(outliers_from_rs)
        outliers_ = outliers_.union(outliers_from_bfr)

        intermediate_results_ = (num_clusters_ds, num_points_ds, num_clusters_cs, num_points_cs, num_points_rs)
        return discard_summary_, outliers_, intermediate_results_

    def write_intermediate_result(output_file_path: str, intermediate_results_: tuple):
        """
        Output the intermediate results during the BFR process to disk
        :param output_file_path: output file path, should be a csv file
        :param intermediate_results_: summary information of the discard, compressed and retained set
        :return: None
        """
        num_clusters_ds, num_points_ds, num_clusters_cs, num_points_cs, num_points_rs = intermediate_results_
        with open(output_file_path, "w") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["round_id", "nof_cluster_discard", "nof_point_discard",
                             "nof_cluster_compression", "nof_point_compression", "nof_point_retained"])
            for i in range(num_files):
                writer.writerow([i + 1, num_clusters_ds[i], num_points_ds[i],
                                 num_clusters_cs[i], num_points_cs[i], num_points_rs[i]])

    def write_clustering_result(output_file_path: str, discard_summary_: dict, outliers: set):
        """
        Output the cluster assignments of every point in the whole dataset
        :param output_file_path: output file path, should be a json file
        :param discard_summary_: summary information of discard set
        :param outliers: point id of outliers_ obtained from standardizing data, leftover compressed set and retained set
        :return: None
        """
        outliers = list(map(lambda x: str(int(x)), outliers))
        output = dict.fromkeys(outliers, -1)
        for cid, points in discard_summary_["assignments"].items():
            point_assignments = dict.fromkeys(list(map(lambda x: str(int(x)), points)), cid)
            output.update(point_assignments)

        output_list = sorted([(key, value) for key, value in output.items()], key=lambda pair: int(pair[0]))
        output = dict(output_list)

        with open(output_file_path, "w+") as file:
            json.dump(output, file)


    start_time = time.time()
    input_file_list = os.listdir(input_path)
    print("input_file_list_", input_file_list)
    num_files = len(input_file_list)

    discard_summary, outliers, intermediate_results = bfr(input_file_list, num_clusters)

    write_intermediate_result(intermediate_file_path, intermediate_results)
    write_clustering_result(clustering_file_path, discard_summary, outliers)
    print("total running time:", time.time() - start_time)


if __name__ == '__main__':
    main(sys.argv)
