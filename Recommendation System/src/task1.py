from pyspark import SparkContext
from pyspark.rdd import RDD
import gc
import sys
import time
import math
import json
import random

SparkContext.setSystemProperty('spark.executor.memory', '7g')
SparkContext.setSystemProperty('spark.driver.memory', '7g')
sc = SparkContext(appName="INF553HW3", master="local[*]")
sc.setLogLevel("WARN")
sc.setLogLevel("ERROR")

def main(argv):
    # input_file_path = argv[1]
    # output_file_path = argv[2]
    input_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW3/data/train_review.json"
    output_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW3/output/task1.res"

    # 26184 users, 10253 business
    total_start = time.time()
    review_data = sc.textFile(input_file_path).map(lambda line: json.loads(line)).persist()
    unique_user_id = review_data. \
        map(lambda line: line['user_id']). \
        distinct(). \
        collect()

    user_id_to_index = {}
    for i in range(len(unique_user_id)):
        user_id_to_index[unique_user_id[i]] = i
    user_id_to_index = sc.broadcast(user_id_to_index)
    num_users = len(user_id_to_index.value)
    index_range = sc.broadcast([i for i in range(num_users)])


    def build_minhash_signatures(feature_values: list, num_hash_functions: int) -> list:
        def hash_function_generator(a, b, m=10000):
            return lambda x: hash(str(a * x + b)) % m

        signatures = []
        for seed in range(num_hash_functions):
            random.seed(seed)
            parameter = random.random()
            index_range_copy = index_range.value
            hashed_values = list(map(hash_function_generator(parameter, parameter),
                                     [index_range_copy[i] for i in feature_values]))
            signature = min(hashed_values)
            signatures.append(signature)
        return signatures

    def hash_signature_to_bucket(signatures: list, num_bands: int, num_rows: int) -> list:
        return [hash(sum(signatures[band * num_rows: (band + 1) * num_rows])) % 3000 for band in range(num_bands)]

    def qualified_as_candidate_pair(buckets_1: list, buckets_2: list) -> bool:
        # for i in range(len(buckets_1)):
        #     if buckets_1[i] == buckets_2[i]:
        #         return True
        # return False
        if len(set(buckets_1).intersection(set(buckets_2))) >= 1:
            return True
        else:
            return False

    def jaccard_similarity(features_1: list, features_2: list) -> float:
        set_1 = set(features_1)
        set_2 = set(features_2)
        similarity = len(set_1.intersection(set_2)) / len(set_1.union(set_2))
        return similarity


    num_hash_functions = 225
    num_bands = 75
    num_rows = 3

    # signature_matrix
    signature_matrix = review_data. \
        map(lambda line: (line['business_id'], line['user_id'])). \
        distinct(). \
        groupByKey(). \
        mapValues(lambda ids: [user_id_to_index.value[id] for id in ids]). \
        mapValues(lambda values: (values, build_minhash_signatures(values, num_hash_functions))).\
        mapValues(lambda pair: (pair[0],
                                hash_signature_to_bucket(pair[1], num_bands, num_rows))). \
        filter(lambda pair: len(pair[1][0]) <= 80).\
        persist()

    start = time.time()
    _ = signature_matrix.first()
    print("finish building signatures and bucket serials:", time.time() - start)

    start = time.time()

    output = signature_matrix.\
        cartesian(signature_matrix). \
        filter(lambda pair: pair[0][0] < pair[1][0]).\
        filter(lambda pair: qualified_as_candidate_pair(pair[0][1][1], pair[1][1][1])). \
        map(lambda pair: (pair[0][0], pair[1][0], jaccard_similarity(pair[0][1][0], pair[1][1][0]))). \
        filter(lambda pair: pair[2] >= 0.05). \
        collect()

    output = sorted(output, key=lambda _: _[2], reverse=True)
    print(len(output))  # 59435 ground-truth, so 0.8 accuracy needs at least 47548
    print("finish:", time.time() - start)  # for loop: 153s/ 111s, set operation: 204s

    with open(output_file_path, "w") as output_file:
        for i in range(len(output)):
            json.dump({"b1": output[i][0], "b2": output[i][1], "sim": output[i][2]}, output_file)
            output_file.write("\n")

    print("total time:", time.time() - total_start)


if __name__ == '__main__':
    main(sys.argv)
