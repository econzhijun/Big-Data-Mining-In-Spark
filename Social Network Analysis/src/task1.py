from pyspark import SparkContext
from pyspark.sql import SparkSession, Row
import sys
import time
from graphframes import GraphFrame

# spark version 2.4.4 Using Scala version 2.11.12
# pyspark  __version__ : 2.4.4
SparkContext.setSystemProperty('spark.executor.memory', '7g')
SparkContext.setSystemProperty('spark.driver.memory', '7g')
sc = SparkContext(appName="INF553HW4", master="local[1]")
sc.setLogLevel("WARN")
sc.setLogLevel("ERROR")
ss = SparkSession.builder.\
    master("local[*]"). \
    appName("INF553HW4"). \
    getOrCreate()

def main(argv):
    filter_threshold = int(argv[1])
    input_file_path = argv[2]
    output_file_path = argv[3]
    # os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.4-s_2.11")

    # filter_threshold = 7
    # input_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW4/data/ub_sample_data.csv"
    # output_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW4/output/task1.txt"

    # 38648 records
    total_start = time.time()
    start = time.time()
    input_data = sc.textFile(input_file_path).\
        filter(lambda line: "user_id" not in line).\
        map(lambda line: tuple(line.split(","))).\
        groupByKey().\
        mapValues(set).\
        persist()  # 3374

    edges = input_data.\
        cartesian(input_data).\
        filter(lambda pair: pair[0][0] < pair[1][0]).\
        filter(lambda pair: len(pair[0][1].intersection(pair[1][1])) >= filter_threshold).\
        flatMap(lambda pair: [(pair[0][0], pair[1][0]), (pair[1][0], pair[0][0])]).\
        persist()  # 996 498
    edges_df = edges.map(lambda pair: Row(src=pair[0], dst=pair[1])).toDF()

    vertices = edges.flatMap(lambda _:_).distinct().persist()  # 222
    vertices_df = vertices.map(Row("id")).toDF()
    print("finish building edges and vertices:", time.time() - start)

    start = time.time()
    graph = GraphFrame(vertices_df, edges_df)
    result = graph.labelPropagation(maxIter=5)
    print("finish running LPA:", time.time() - start)
    # result.count()  # 222
    # result.show()

    result_rdd = result.rdd.\
        map(lambda pair: (pair['label'], pair['id'])).\
        groupByKey().\
        mapValues(lambda values: (sorted(list(values)), len(values))).\
        persist()

    result_collection = result_rdd.collect()
    result_collection.sort(key=lambda kv: (kv[1][1], kv[1][0][0]))
    with open(output_file_path, "w") as output_file:
        for community_id, (user_list, length) in result_collection:
            output_file.write(f"'{user_list[0]}'")
            for user in user_list[1:]:
                output_file.write(f", '{user}'")
            output_file.write("\n")
    print("total running time:", time.time() - total_start)


if __name__ == '__main__':
    main(sys.argv)