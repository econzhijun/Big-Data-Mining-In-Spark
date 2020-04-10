from pyspark import SparkContext
import json
import sys
import gc
import time

SparkContext.setSystemProperty('spark.executor.memory', '5g')
SparkContext.setSystemProperty('spark.driver.memory', '5g')
sc = SparkContext(appName="INF553HW1", master="local[*]")

def main(argv):
    review_file_path = argv[1]
    out_file_path = argv[2]
    partition_type = argv[3]
    n = int(argv[5])

    assert partition_type in ["default", "customized"]
    # review_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW1/data/review.json"
    review_data = sc.textFile(review_file_path).map(lambda line: json.loads(line)).persist()

    if partition_type == "default":
        # start = time.time()
        business_qualified = review_data.map(lambda line: (line["business_id"], 1)).\
            reduceByKey(lambda x, y: x+y).filter(lambda pair: pair[1] > n).collect()
        business_qualified = [list(_) for _ in business_qualified]
        n_partitions = review_data.getNumPartitions()
        n_items_list = review_data.mapPartitions(lambda partition: [sum(1 for _ in partition)]).collect()
        # end = time.time()
        # print("time:", end - start)

    if partition_type == "customized":
        def custom_partitioner(value):
            return hash(value)

        n_partitions = int(argv[4])
        # start = time.time()
        review_data_customed = review_data.map(lambda e: (e['business_id'], 1)).\
            partitionBy(numPartitions=n_partitions, partitionFunc=custom_partitioner).persist()
        business_qualified = review_data_customed.reduceByKey(lambda x, y: x+y).filter(lambda pair: pair[1] > n).collect()
        business_qualified = [list(_) for _ in business_qualified]
        n_items_list = review_data_customed.mapPartitions(lambda partition: [sum(1 for _ in partition)]).collect()
        # end = time.time()
        # print("time:", end - start)

    result = {"n_partitions": n_partitions,
              "n_items": n_items_list,
              "result": business_qualified}
    with open(out_file_path, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    main(sys.argv)