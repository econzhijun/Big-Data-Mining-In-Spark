from pyspark import SparkContext
from pyspark import SparkConf
import json
import sys
import gc
import time

SparkContext.setSystemProperty('spark.executor.memory', '5g')
SparkContext.setSystemProperty('spark.driver.memory', '5g')
sc = SparkContext(appName="INF553HW1", master="local[*]")

scf = SparkConf().setAppName("INF553-HW1-Task1").setMaster("local[*]").set('spark.driver.memory', '5g')
sc = SparkContext.getOrCreate(conf=scf)

# # def timed(expression):
# start = time.time()
# # result = expression
# # total_num_review = review_data.aggregate(0, seqOp, combOp)
# review_data.count()
# end = time.time()
# print("used time is:", end-start)
#     # return result


def main(argv):
    review_file_path = argv[1]
    out_file_path = argv[2]
    stopwords_file_path = argv[3]
    y = int(argv[4])
    m = int(argv[5])
    n = int(argv[6])

    # stopwords_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW1/data/stopwords"
    # review_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW1/data/review.json"
    # y = 2018
    # m = 20
    # n = 20

    review_data = sc.textFile(review_file_path).map(lambda line: json.loads(line)).persist()

    # // part A
    total_num_review = review_data.count()
    print("total_num_review", total_num_review)  # 1151625

    # // part B
    total_num_review_in_year_y = review_data.filter(lambda line: int(line['date'][:4]) == y).count()
    print("total_num_review_in_year_y", total_num_review_in_year_y)  # 209995

    # // part C
    # review_samples.map(lambda line: line['stars']).distinct().count()
    distinct_users = review_data.map(lambda line: line['user_id']).distinct().count()
    print("distinct_users", distinct_users)

    # // part D
    top_m_users = review_data.map(lambda line: (line['user_id'], 1)).\
        reduceByKey(lambda x, y: x+y, numPartitions=4).\
        sortBy(keyfunc=lambda pair: (pair[1], pair[0]), ascending=False, numPartitions=4).\
        take(m)
    top_m_users = [list(pair) for pair in top_m_users]

    # // part E
    stopwords = open(stopwords_file_path, "r").readlines()
    stopwords = [word.rstrip("\n") for word in stopwords]
    stopwords.append("")
    punctuations = ["(", "[", ",", ".", "!", "?", ":", ";", "]", ")", "\n", "\\"]

    def exclude_punctuations(text: str):
        for punctuation in punctuations:
            text = text.replace(punctuation, " ")
        return text

    top_n_words = review_data.\
        flatMap(lambda line: exclude_punctuations(line["text"].lower()).split(" ")).\
        filter(lambda word: word not in stopwords).\
        map(lambda word: (word, 1)).\
        reduceByKey(lambda x, y: x+y, numPartitions=4).\
        sortBy(keyfunc=lambda pair: (pair[1], pair[0]), ascending=False, numPartitions=4).\
        take(n)
    top_n_words = [pair[0] for pair in top_n_words]

    result = {"A": total_num_review,
              "B": total_num_review_in_year_y,
              "C": distinct_users,
              "D": top_m_users,
              "E": top_n_words}

    with open(out_file_path, 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    main(sys.argv)