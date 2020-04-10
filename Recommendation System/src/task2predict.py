from pyspark import SparkContext
from pyspark.rdd import RDD
import gc
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
    total_time = time.time()

    test_file_path = argv[1]
    model_file_path = argv[2]
    output_file_path = argv[3]
    # test_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW3/data/test_review.json"
    # output_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW3/output/task2.predict"
    # model_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW3/output/task2.model"

    def preprocess_data(line: str) -> dict:
        "this is used when using method RDD.saveAsTextFile in the training phase"
        result = line.replace("\'", "\"").replace(" {", ' "{').replace("}}", '}" }').replace("},", '}",')
        return json.loads(result)

    def get_business_data(sample):
        try:
            _ = sample['business_id']
            return True
        except KeyError:
            return False

    def get_user_data(sample):
        try:
            _ = sample['user_id']
            return True
        except:
            return False

    def filter_user_id(user_id):
        try:
            _ = user_id_in_test.value[user_id]
            return True
        except KeyError:
            return False

    def filter_business_id(business_id):
        try:
            _ = business_id_in_test.value[business_id]
            return True
        except KeyError:
            return False

    def check_in_dictionary(key: str, dictionary: dict) -> bool:
        try:
            _ = dictionary[key]
            return True
        except KeyError:
            return False

    def cosine_similarity(user_profile: dict, user_profile_norm: float, business_profile: dict) -> float:
        dot_product = 0
        for key in business_profile.keys():
            try:
                dot_product += user_profile[key]
            except KeyError:
                continue
        return dot_product / (business_profile_norm.value * user_profile_norm)

    # process data
    start = time.time()
    test_data = sc.textFile(test_file_path).\
        map(lambda line: json.loads(line)).\
        map(lambda line: (line['user_id'], line['business_id'])).\
        persist()  # 58480
    user_id_in_test = sc.broadcast(dict.fromkeys(test_data.map(lambda _: _[0]).distinct().collect()))
    business_id_in_test = sc.broadcast(dict.fromkeys(test_data.map(lambda _: _[1]).distinct().collect()))

    model_file = sc.textFile(model_file_path).map(preprocess_data).persist()  # 36437
    business_profiles_data = model_file.filter(get_business_data).\
        filter(lambda line: filter_business_id(line['business_id'])).\
        map(lambda line: (line['business_id'], eval(line['business_profile'])))  # 9211
    business_profiles_dict = sc.broadcast(dict(business_profiles_data.collect()))
    business_profile_norm = sc.broadcast(math.sqrt(200))

    user_profiles_data = model_file.\
        filter(lambda line: get_user_data(line)).\
        filter(lambda line: filter_user_id(line['user_id'])).\
        map(lambda line: (line['user_id'], (eval(line['user_profile']), line['user_profile_norm']))).\
        persist()  # 20811
        # map(lambda line: (line['user_id'], ( line['user_profile'], line['user_profile_norm']))). \

    print("finish processing:", time.time() - start)  # 74.665s

    test_data: RDD = user_profiles_data.\
        join(test_data). \
        filter(lambda pair: check_in_dictionary(key=pair[1][1],
                                                dictionary=business_profiles_dict.value)). \
        persist()  # 58473 58480

    output = test_data.\
        map(lambda pair: ( (pair[0], pair[1][1]),
                           cosine_similarity(user_profile=pair[1][0][0],
                                             user_profile_norm=pair[1][0][1],
                                             business_profile=business_profiles_dict.value[pair[1][1]]))).\
        filter(lambda pair: pair[1] >= 0.01).\
        map(lambda pair: {"user_id": pair[0][0], "business_id": pair[0][1], "sim": round(pair[1], 5)}).\
        persist()

    start = time.time()
    output_result_collection = output.collect()
    output_result_collection = sorted(output_result_collection, key=lambda _: _['sim'], reverse=True)
    with open(output_file_path, "w") as output_file:
        for i in range(len(output_result_collection)):
            json.dump(output_result_collection[i], output_file, separators=(', ', ': '))
            output_file.write("\n")

    print("finish computing similarity and writing: ", time.time() - start)  # 17.401s
    print("total running time:", time.time() - total_time)


if __name__ == '__main__':
    main(sys.argv)
