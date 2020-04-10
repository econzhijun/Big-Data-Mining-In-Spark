from pyspark import SparkContext
from pyspark.rdd import RDD
from pyspark.resultiterable import ResultIterable
import sys
import time
import math
import json

SparkContext.setSystemProperty('spark.executor.memory', '7g')
SparkContext.setSystemProperty('spark.driver.memory', '7g')
sc = SparkContext(appName="INF553HW3", master="local[*]")
sc.setLogLevel("WARN")
sc.setLogLevel("ERROR")


def main(argv):
    train_file_path = argv[1]
    model_file_path = argv[2]
    cf_type = argv[3]
    assert cf_type in ["item_based", "user_based"]

    train_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW3/data/train_review.json"
    model_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW3/output/task3user.model"
    cf_type = "user_based"

    total_start = time.time()

    # def get_profile_norm(profile: dict) -> float:
    #     return math.sqrt(sum([v ** 2 for v in profile.values()]))
    #

    # def normalize_profile(profile: dict) -> dict:
    #     average = sum([value for value in profile.values()]) / len(profile)
    #     return dict([(key, value - average) for key, value in profile.items()])

    # def cosine_similarity(features_1: dict, features_2: dict, norm_1: float, norm_2: float) -> float:
    #     "We should traverse the shorter one"
    #     if len(features_1) > len(features_2):
    #         features_1, features_2 = features_2, features_1
    #
    #     dot_product = 0
    #     for key in features_1.keys():
    #         try:
    #             dot_product += features_2[key] * features_1[key]
    #         except:
    #             continue
    #     return dot_product / (norm_1 * norm_2)

    def qualified_as_candidate_pair(profile_1: dict, profile_2: dict) -> bool:
        if len(set(profile_1).intersection(set(profile_2))) >= 3:
            return True
        else:
            return False

    def pearson_correlation(features_1: dict, features_2: dict) -> float:
        intersection = list(set(features_1).intersection(set(features_2)))
        intersection_1 = [features_1[key] for key in intersection]
        intersection_2 = [features_2[key] for key in intersection]
        avg_1 = sum(intersection_1) / len(intersection_1)
        avg_2 = sum(intersection_2) / len(intersection_2)
        intersection_1 = [value - avg_1 for value in intersection_1]
        intersection_2 = [value - avg_2 for value in intersection_2]
        nominator = sum([intersection_1[j] * intersection_2[j] for j in range(len(intersection))])
        norm_1 = math.sqrt(sum([v ** 2 for v in intersection_1]))
        norm_2 = math.sqrt(sum([v ** 2 for v in intersection_2]))
        # return nominator / (norm_1 * norm_2)
        try:
            scale = 1 / (norm_1 * norm_2)
            return nominator * scale
        except ZeroDivisionError:
            return -1

    train_data = sc.textFile(train_file_path).map(lambda line: json.loads(line)).persist()

    if cf_type == "item_based":
        unique_user_id = train_data. \
            map(lambda line: line['user_id']). \
            distinct(). \
            collect()
        user_id_to_index = {}
        for i in range(len(unique_user_id)):
            user_id_to_index[unique_user_id[i]] = i
        user_id_to_index = sc.broadcast(user_id_to_index)

        signature_matrix = train_data. \
            map(lambda line: (line['business_id'], (line['user_id'], line['stars']))). \
            mapValues(lambda pair: (user_id_to_index.value[pair[0]], pair[1])). \
            groupByKey(). \
            mapValues(lambda values: dict(values)). \
            persist()
    else:
        unique_business_id = train_data. \
            map(lambda line: line['business_id']). \
            distinct(). \
            collect()
        business_id_to_index = {}
        for i in range(len(unique_business_id)):
            business_id_to_index[unique_business_id[i]] = i
        business_id_to_index = sc.broadcast(business_id_to_index)

        signature_matrix = train_data. \
            map(lambda line: (line['user_id'], (line['business_id'], line['stars']))). \
            mapValues(lambda pair: (business_id_to_index.value[pair[0]], pair[1])). \
            groupByKey(). \
            mapValues(lambda values: dict(values)). \
            persist()

    pair_similarities = signature_matrix. \
        cartesian(signature_matrix). \
        filter(lambda pair: pair[0][0] < pair[1][0]). \
        filter(lambda pair: qualified_as_candidate_pair(pair[0][1], pair[1][1])). \
        map(lambda pair: ((pair[0][0], pair[1][0]),
                          pearson_correlation(features_1=pair[0][1], features_2=pair[1][1]))). \
        filter(lambda pair: pair[1] > 0). \
        persist()

    start = time.time()
    print("collecting model outputs")
    model_outputs = pair_similarities.collect()
    print("start writing file")
    if cf_type == "item_based":
        with open(model_file_path, "w") as output_file:
            for i in range(len(model_outputs)):
                output = model_outputs[i]
                json.dump({"b1": output[0][0], "b2": output[0][1], "sim": output[1]}, output_file)
                output_file.write("\n")
    else:
        with open(model_file_path, "w") as output_file:
            for i in range(len(model_outputs)):
                output = model_outputs[i]
                json.dump({"u1": output[0][0], "u2": output[0][1], "sim": output[1]}, output_file)
                output_file.write("\n")
    print("run time:", time.time() - start)

    print("total runtime:", time.time() - total_start)
    # item_based 91.725s vocareum
    # user_based 250.89s local, 448.67s vocareum

if __name__ == '__main__':
    main(sys.argv)
