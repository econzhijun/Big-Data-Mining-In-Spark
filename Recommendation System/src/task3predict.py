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
    train_file_path = argv[1]
    test_file_path = argv[2]
    model_file_path = argv[3]
    output_file_path = argv[4]
    cf_type = argv[5]
    assert cf_type in ["item_based", "user_based"]

    # train_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW3/data/train_review.json"
    # test_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW3/data/test_review.json"
    # model_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW3/output/task3user.model"
    # output_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW3/output/task3user.predict"
    # cf_type = "user_based"

    # def get_similar_business(user_id: str, business_id: str) -> list:
    #     try:
    #         _ = model_dict.value[business_id]
    #         return _
    #     except:
    #         return [("average", user_average_rating.value[user_id])]

    # def get_user_business_rating(user_id: str, business_id: str) -> float:
    #     try:
    #         stars = user_business_rating_dict.value[(user_id, business_id)]
    #         return stars
    #     except:
    #         return None

    def get_business_similarity_rating_pairs(target_business: str, business_rating_pairs: list,
                                             top_similar: int = 10, normalized=False) -> list:
        similarity_rating_pairs = []
        for business_id, rating in business_rating_pairs:
            try:
                similarity = item_based_model_dict.value[tuple(sorted((business_id, target_business)))]
                if normalized:
                    try:
                        business_average = business_average_rating.value[business_id]
                    except KeyError:
                        business_average = business_average_rating.value["unknown"]
                    similarity_rating_pairs.append((similarity, rating - business_average))
                else:
                    similarity_rating_pairs.append((similarity, rating))
            except KeyError:
                continue
        if similarity_rating_pairs == []:
            if normalized:
                return [(1, 0)]
            else:
                try:
                    business_average = business_average_rating.value[target_business]
                    return [(1, business_average)]
                except KeyError:
                    return [(1, business_average_rating.value['unknown'])]
        else:
            similarity_rating_pairs.sort(key=lambda kv: kv[0], reverse=True)
            return similarity_rating_pairs[:top_similar]

    def get_user_similarity_rating_pairs(target_user: str, user_rating_pairs: list,
                                         top_similar: int = 10, normalized=False) -> list:
        similarity_rating_pairs = []
        for user_id, rating in user_rating_pairs:
            try:
                similarity = user_based_model_dict.value[tuple(sorted((user_id, target_user)))]
                if normalized:
                    try:
                        user_average = user_average_rating.value[user_id]
                    except KeyError:
                        user_average = user_average_rating.value['unknown']
                    similarity_rating_pairs.append((similarity, rating - user_average))
                else:
                    similarity_rating_pairs.append((similarity, rating))
            except KeyError:
                continue
        if not similarity_rating_pairs:
            if normalized:
                return [(1, 0)]
            else:
                try:
                    user_average = user_average_rating.value[target_user]
                    return [(1, user_average)]
                except KeyError:
                    return [(1, user_average_rating.value['unknown'])]
        else:
            similarity_rating_pairs.sort(key=lambda kv: kv[0], reverse=True)
            return similarity_rating_pairs[:top_similar]

    def weighted_rating(similarity_rating_pairs: list) -> float:
        total_sum = 0
        weight_sum = 0
        for similarity, rating in similarity_rating_pairs:
            total_sum += rating * similarity
            weight_sum += similarity
        return total_sum / weight_sum

    total_start = time.time()
    if cf_type == "item_based":
        item_based_model = sc.textFile(model_file_path). \
            map(lambda line: json.loads(line)). \
            map(lambda line: ((line['b1'], line['b2']), line['sim']))
        item_based_model_dict = sc.broadcast(dict(item_based_model.collect()))

        business_average_rating = sc.textFile(train_file_path). \
            map(lambda line: json.loads(line)). \
            map(lambda line: (line['business_id'], (line['stars'], 1))). \
            reduceByKey(lambda this, that: (this[0] + that[0], this[1] + that[1])). \
            mapValues(lambda values: values[0] / values[1]). \
            collect()
        business_average_rating = dict(business_average_rating)
        business_average_rating["unknown"] = sum(business_average_rating.values()) / len(business_average_rating)
        business_average_rating = sc.broadcast(business_average_rating)

        user_business_rating = sc.textFile(train_file_path). \
            map(lambda line: json.loads(line)). \
            map(lambda line: (line["user_id"], (line["business_id"], line['stars']))). \
            groupByKey(). \
            collect()
        user_business_rating_dict = sc.broadcast(dict(user_business_rating))

        def get_final_prediction(weighted_average: float, business_id: str, normalized=False):
            if normalized:
                try:
                    business_average = business_average_rating.value[business_id]
                except KeyError:
                    business_average = business_average_rating.value['unknown']
                return weighted_average + business_average
            else:
                return weighted_average

        test_data = sc.textFile(test_file_path). \
            map(lambda line: json.loads(line)). \
            map(lambda line: ((line['user_id'], line['business_id']),
                              list(user_business_rating_dict.value[line['user_id']]))). \
            map(lambda pair: (pair[0], get_business_similarity_rating_pairs(target_business=pair[0][1],
                                                                            business_rating_pairs=pair[1],
                                                                            top_similar=50,
                                                                            normalized=False))). \
            mapValues(lambda values: weighted_rating(values)). \
            map(lambda pair: (pair[0], get_final_prediction(weighted_average=pair[1],
                                                            business_id=pair[0][1],
                                                            normalized=False))). \
            persist()
    else:
        user_based_model = sc.textFile(model_file_path). \
            map(lambda line: json.loads(line)). \
            map(lambda line: ((line['u1'], line['u2']), line['sim']))
        user_based_model_dict = sc.broadcast(dict(user_based_model.collect()))

        user_average_rating = sc.textFile(train_file_path). \
            map(lambda line: json.loads(line)). \
            map(lambda line: (line['user_id'], (line['stars'], 1))). \
            reduceByKey(lambda this, that: (this[0] + that[0], this[1] + that[1])). \
            mapValues(lambda values: values[0] / values[1]). \
            collect()
        user_average_rating = dict(user_average_rating)
        user_average_rating['unknown'] = sum(user_average_rating.values()) / len(user_average_rating)
        user_average_rating = sc.broadcast(user_average_rating)

        business_user_rating = sc.textFile(train_file_path). \
            map(lambda line: json.loads(line)). \
            map(lambda line: (line["business_id"], (line["user_id"], line['stars']))). \
            groupByKey(). \
            collect()
        business_user_rating_dict = sc.broadcast(dict(business_user_rating))

        def get_business_user_rating(business_id: str) -> list:
            try:
                business_user_ratings = list(business_user_rating_dict.value[business_id])
                return business_user_ratings
            except KeyError:
                return []

        def get_final_prediction(weighted_average: float, user_id: str, normalized=False):
            if normalized:
                try:
                    user_average = user_average_rating.value[user_id]
                except KeyError:
                    user_average = user_average_rating.value['unknown']
                return weighted_average + user_average
            else:
                return weighted_average

        test_data = sc.textFile(test_file_path). \
            map(lambda line: json.loads(line)). \
            map(lambda line: ((line['user_id'], line['business_id']),
                              get_business_user_rating(line["business_id"]))). \
            map(lambda pair: (pair[0], get_user_similarity_rating_pairs(target_user=pair[0][0],
                                                                        user_rating_pairs=pair[1],
                                                                        top_similar=200,
                                                                        normalized=True))). \
            mapValues(lambda values: weighted_rating(values)). \
            map(lambda pair: (pair[0], get_final_prediction(pair[1], pair[0][0],
                                                            normalized=True))). \
            persist()

        test_data.take(20)

    print("start writing file")
    predict_outputs = test_data.collect()
    with open(output_file_path, "w") as output_file:
        for i in range(len(predict_outputs)):
            output = predict_outputs[i]
            json.dump({"user_id": output[0][0], "business_id": output[0][1], "stars": output[1]}, output_file)
            output_file.write("\n")
    print("total time:", time.time() - total_start)
    # item_based 22s vocareum
    # user_based 34s vocareum


if __name__ == '__main__':
    main(sys.argv)
