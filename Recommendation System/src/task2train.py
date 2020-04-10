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

    review_file_path = argv[1]
    model_file_path = argv[2]
    stopwords_file_path = argv[3]

    # 26184 users, 10253 business
    # review_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW3/data/train_review.json"
    # stopwords_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW3/data/stopwords"
    # model_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW3/output/task2.model"

    stopwords = open(stopwords_file_path, "r").readlines()
    stopwords = [word.rstrip("\n") for word in stopwords]
    stopwords.append("")

    punctuations = ["(", "[", ",", ".", "!", "?", ":", ";", "]", ")", "\n", "\\"]

    def exclude_punctuations(text: str) -> str:
        for punctuation in punctuations:
            text = text.replace(punctuation, " ")
        return text

    review_data = sc.textFile(review_file_path, minPartitions=10).map(lambda line: json.loads(line)).persist()

    documents = review_data.\
        flatMap(lambda line: [(line['business_id'], word)
                              for word in exclude_punctuations(line["text"].lower()).split(" ")]).\
        filter(lambda pair: pair[1] not in stopwords).\
        groupByKey().\
        persist()

    start = time.time()
    num_documents = sc.broadcast(documents.count())  # 10253 businesses
    # 44.884s try to optimize, how come is even slower??
    # 44.598s without removing punctuations
    print("runtime for parsing review documents: ", time.time() - start)  # 51.216s

    start = time.time()
    unique_words = documents.\
        map(lambda pair: pair[1]).\
        flatMap(lambda _: _).\
        distinct().collect()
    # len(unique_words)  # 334861
    unique_words = sc.broadcast(dict.fromkeys(unique_words))
    print("got unique words: ", time.time() - start)  # 10.31s

    # get IDF score
    def contains_words(document: list, unique_words: dict) -> list:
        result = []
        for word in set(document):  # in case one word appears in this document more than once
            try:
                _ = unique_words[word]
                result.append((word, 1))
            except:
                continue
        return result

    start = time.time()
    idf = documents.\
        map(lambda pair: pair[1]).\
        flatMap(lambda document: contains_words(document, unique_words.value)).\
        reduceByKey(lambda x, y: x + y).\
        mapValues(lambda count: math.log2(num_documents.value / count))
    idf_score = sc.broadcast(dict(idf.collect()))
    print("got IDF score: ", time.time() - start)  # 9.407s hash table赛高！效果看得见！

    # get TF-IDF score
    def calculate_tf_within_document(word_count_pairs: list):
        max_count = max(word_count_pairs, key=lambda kv: kv[1])[1]
        return [(word, count / max_count) for word, count in word_count_pairs]

    start = time.time()
    tf_idf = documents.\
        flatMap(lambda pair: [((pair[0], word), 1) for word in pair[1]]).\
        reduceByKey(lambda x, y: x+y).\
        map(lambda pair: (pair[0][0], (pair[0][1], pair[1]))).\
        groupByKey().\
        mapValues(lambda values: calculate_tf_within_document(values)).\
        mapValues(lambda values: [(word, tf * idf_score.value[word]) for word, tf in values]).\
        mapValues(lambda values: sorted(values, key=lambda kv: kv[1], reverse=True)[:200]). \
        persist()

    unique_tf_idf_words = tf_idf.\
        flatMap(lambda pair: pair[1]).\
        map(lambda pair: pair[0]).\
        distinct().\
        collect()  # 141965 words  # 变成262158了？  # 215173 增加了list(set(操作之后变了

    tf_idf_words_index = {}
    for i in range(len(unique_tf_idf_words)):
        tf_idf_words_index[unique_tf_idf_words[i]] = i
    tf_idf_words_index = sc.broadcast(tf_idf_words_index)
    print("runtime get TF-IDF score: ", time.time() - start)  # 47.267s # 60.503  74.867s finished collect
    del documents
    gc.collect()

    # build Business profiles
    def build_business_profile_vector(top_tf_idf_words) -> dict:
        return dict([(tf_idf_words_index.value[word], 1) for word, score in top_tf_idf_words])

    business_profiles_data = tf_idf.\
        mapValues(lambda values: build_business_profile_vector(values)).\
        persist()  # format: (business_id, business_profile)
    # business_profiles_collection = business_profiles_data.collect()
    del tf_idf
    gc.collect()

    # build User profiles
    utility_matrix = review_data.\
        map(lambda line: (line['business_id'], line['user_id'])).\
        distinct().\
        groupByKey()
    utility_matrix_dict = sc.broadcast(dict(utility_matrix.collect()))  # 26184 users, 10253 business
    del review_data
    gc.collect()

    def assign_profile_to_user(business_profile_pair: tuple) -> list:
        "business_profile_pair is (business_id, business_profile) tuple"
        try:
            reviewed_by_users = utility_matrix_dict.value[business_profile_pair[0]]
            return [(user, business_profile_pair[1]) for user in reviewed_by_users ]
        except:
            return [()]

    def get_aggregated_profile(business_profiles: list) -> dict:
        "take average of the profiles belonging to this user"
        length = len(business_profiles)
        aggregated_profile = {}
        for business_profile in business_profiles:
            for i in business_profile.keys():
                aggregated_profile[i] = aggregated_profile.get(i, 0) + 1 / length
        return aggregated_profile

    def get_profile_norm(profile: dict) -> float:
        return math.sqrt(sum([v**2 for v in profile.values()]))


    user_profiles_data = business_profiles_data.\
        flatMap(lambda pair: assign_profile_to_user(pair)).\
        filter(lambda pair: pair[1] != ()).\
        groupByKey().\
        mapValues(lambda v: get_aggregated_profile(v.data)).\
        map(lambda pair: {"user_id": pair[0],
                          "user_profile": pair[1],
                          "user_profile_norm": get_profile_norm(pair[1])}).\
        persist()  # format: (user_id, (user_profile, user_profile_norm))
    # user_profiles_collection = user_profiles_data.collect()

    start = time.time()
    model = business_profiles_data.\
        map(lambda pair: {"business_id": pair[0], "business_profile": pair[1]}).\
        union(user_profiles_data)
    model.saveAsTextFile(model_file_path)
    print("saved model: ", time.time() - start)  # 20.762s

    print("total running time:", time.time() - total_time)  # 156.652s in total  # 142s on vocareum


if __name__ == '__main__':
    main(sys.argv)