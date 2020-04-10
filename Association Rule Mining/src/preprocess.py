from pyspark import SparkContext
import json
import gc
import csv

SparkContext.setSystemProperty('spark.executor.memory', '7g')
SparkContext.setSystemProperty('spark.driver.memory', '7g')
sc = SparkContext(appName="INF553HW2", master="local[*]")

# review_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW2/data/review.json"
# business_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW2/data/business.json"

def preprocess(business_file_path, review_file_path, csv_file_path):
    business_id = sc.textFile(business_file_path).\
        map(lambda line: json.loads(line)).\
        filter(lambda line: line['state'] == "NV").\
        map(lambda line: (line["business_id"], 1)).collect()
    business_id_dict = dict(business_id)
    del business_id
    gc.collect()

    def check_business_id_in_nv(id):
        try:
            _ = business_id_dict[id]
            return True
        except:
            return False

    review_data = sc.textFile(review_file_path).\
        map(lambda line: json.loads(line)).\
        filter(lambda line: check_business_id_in_nv(line['business_id'])).\
        map(lambda line: (line['user_id'], line['business_id'])).\
        persist()   # 6685900 lines in total

    # review_data.count()  # 2320491 lines
    with open(csv_file_path, 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(['user_id', 'business_id'])
        for row in review_data.collect():
            writer.writerow(row)
