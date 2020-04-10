from pyspark import SparkContext
from pyspark import SparkConf
import json
import sys
import gc
import time

SparkContext.setSystemProperty('spark.executor.memory', '5g')
SparkContext.setSystemProperty('spark.driver.memory', '5g')
sc = SparkContext(appName="INF553HW1", master="local[*]")
sc.setLogLevel("ERROR")


def main(argv):
    review_file_path = argv[1]
    business_file_path = argv[2]
    out_file_path = argv[3]
    if_spark = argv[4]
    n = int(argv[5])

    if if_spark == "spark":
        review_data = sc.textFile(review_file_path).map(lambda line: json.loads(line))
        business_data = sc.textFile(business_file_path).map(lambda line: json.loads(line))

        top_n_categories = business_data. \
            filter(lambda line: line['categories'] is not None). \
            map(lambda line: (line['business_id'], line['categories'])). \
            join(review_data.map(lambda line: (line['business_id'], line['stars']))). \
            flatMap(lambda line: [(category.strip(), line[1][1]) for category in line[1][0].split(",")]). \
            aggregateByKey(zeroValue=(0, 0),
                           seqFunc=lambda accumulate, value: (accumulate[0] + value, accumulate[1] + 1),
                           combFunc=lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])). \
            mapValues(lambda pair: pair[0] / pair[1]). \
            sortBy(keyfunc=lambda pair: (-pair[1], pair[0]), ascending=True, numPartitions=4). \
            take(n)
        top_n_categories = [list(_) for _ in top_n_categories]

    if if_spark == "no_spark":
        def read_json_to_dict(file_path):
            with open(file_path, 'r') as file:
                data = file.read()
                new_data = data.replace("}\n{", '},{')
                json_data = json.loads(f'[{new_data}]')
                del data, new_data
                gc.collect()
            return json_data

        business_python = read_json_to_dict(business_file_path)
        review_python = read_json_to_dict(review_file_path)

        business_python = [{business['business_id']: business['categories']} for business in business_python]
        review_python = [{review['business_id']: review['stars']} for review in review_python]

        business_dicts = {}  # this is one single dict containing lots of key-value pairs
        for d in business_python:
            business_dicts.update(d)
        del business_python
        gc.collect()

        category_stars_sum = {}
        category_stars_count = {}
        for review in review_python:  # get a dict
            business_id = next(iter(review))  # get the key of it
            stars = review[business_id]  # get the value of it
            try:
                categories = business_dicts[business_id]
                if categories is not None:
                    for category in map(lambda s: s.strip(), categories.split(",")):
                        category_stars_sum[category] = category_stars_sum.get(category, 0) + stars
                        category_stars_count[category] = category_stars_count.get(category, 0) + 1
            except:
                continue

        category_stars_average = {}
        for category in category_stars_sum:
            category_stars_average[category] = category_stars_sum[category] / category_stars_count[category]
        top_n_categories = [list(_) for _ in sorted(category_stars_average.items(),
                                                    key=lambda kv: (-kv[1], kv[0]), reverse=False)[:n]]

    result = {"result": top_n_categories}
    with open(out_file_path, 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    main(sys.argv)

