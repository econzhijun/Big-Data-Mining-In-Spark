from pyspark import SparkContext
from pyspark.rdd import RDD
import json
import gc
import sys
import time
import math
import csv

SparkContext.setSystemProperty('spark.executor.memory', '7g')
SparkContext.setSystemProperty('spark.driver.memory', '7g')
sc = SparkContext(appName="INF553HW2", master="local[*]")


def main(argv):
    filter_threshold = int(argv[1])
    support_threshold_total = int(argv[2])
    input_file_path = argv[3]
    output_file_path = argv[4]

    # filter_threshold = 70
    # support_threshold_total = 50
    # input_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW2/data/user_business.csv"

    def generate_combination(input_array: list, size):
        '''
        :param input_array: ["A", "B", "C", "D", "E"]
        :param size: 3
        :return: [['A', 'B', 'C'],
                 ['A', 'B', 'D'],
                 ['A', 'B', 'E'],
                 ['A', 'C', 'D'],
                 ['A', 'C', 'E'],
                 ['A', 'D', 'E'],
                 ['B', 'C', 'D'],
                 ['B', 'C', 'E'],
                 ['B', 'D', 'E'],
                 ['C', 'D', 'E']]
        '''

        def generate_combination_helper(input_array: list, temporary_array: list, size, combinations,
                                        start, end, temporary_array_index):
            if temporary_array_index == size:
                combinations.append(temporary_array.copy())
                return

            input_array_index: int = start
            while (input_array_index <= end):
                temporary_array[temporary_array_index] = input_array[input_array_index]
                generate_combination_helper(input_array, temporary_array, size, combinations,
                                            input_array_index + 1, end, temporary_array_index + 1)
                input_array_index += 1

        combinations = []  # store all the possible combinations
        temporary_array = [-1] * size  # Temporary array to store current combination
        n = len(input_array)
        generate_combination_helper(input_array, temporary_array, size, combinations,
                                    start=0, end=n - 1, temporary_array_index=0)
        return combinations


    def generate_combination_with_filter(input_array: list, frequent_itemsets: dict, size: int) -> list:
        '''
        :param input_array: ["A", "B", "C", "D", "E"]
        :param size: 3
        :return: [['A', 'B', 'C'],
                 ['A', 'B', 'D'],
                 ['A', 'B', 'E'],
                 ['A', 'C', 'D'],
                 ['A', 'C', 'E'],
                 ['A', 'D', 'E'],
                 ['B', 'C', 'D'],
                 ['B', 'C', 'E'],
                 ['B', 'D', 'E'],
                 ['C', 'D', 'E']]
        '''
        def generate_combination_helper(input_array: list, temporary_array: list, size, combinations,
                                        start, end, temporary_array_index):
            # print("temporary_array_index", temporary_array_index)
            # print("temporary_array", temporary_array)
            if temporary_array_index == size:
                for i in range(size-2):
                    try:
                        _ = frequent_itemsets[tuple(temporary_array[-2 - i:])]
                    except:
                        return
                combinations.append(temporary_array.copy())
                # print("save, temporary_array", temporary_array)
                return


            input_array_index: int = start
            early_exit = False

            if temporary_array_index == size - 1:
                while (input_array_index <= end):
                    temporary_array[temporary_array_index] = input_array[input_array_index]
                    try:
                        _ = frequent_itemsets[input_array[input_array_index]]
                    except:
                        # print("singleton exit, temporary_array:", temporary_array, "input_array_index:", input_array_index)
                        input_array_index += 1
                        continue

                    for i in range(1, temporary_array_index):
                        # print("Run, temporary_array_index", temporary_array_index, "i:", i,
                        #       "input_array_index:", input_array_index,
                        #       "temporary_array:", temporary_array)
                        try:
                            # print("tuple:", tuple(sorted(temporary_array[i: temporary_array_index+1])))
                            _ = frequent_itemsets[tuple(temporary_array[i: temporary_array_index+1])]
                        except:
                            # print("except")
                            early_exit = True
                            break

                    if early_exit:
                        input_array_index += 1
                        # print("larger exit, temporary_array:", temporary_array)
                        early_exit = False
                        continue

                    # temporary_array[temporary_array_index] = input_array[input_array_index]
                    generate_combination_helper(input_array, temporary_array, size, combinations,
                                                input_array_index + 1, end, temporary_array_index + 1)
                    input_array_index += 1
            else:
                while (input_array_index <= end):
                    temporary_array[temporary_array_index] = input_array[input_array_index]
                    try:
                        _ = frequent_itemsets[input_array[input_array_index]]
                    except:
                        # print("singleton exit, temporary_array:", temporary_array, "input_array_index:",
                        #       input_array_index)
                        input_array_index += 1
                        continue

                    for i in range(0, temporary_array_index):
                        # print("Run, temporary_array_index", temporary_array_index, "i:", i,
                        #       "input_array_index:", input_array_index,
                        #       "temporary_array:", temporary_array)
                        try:
                            # print("tuple:", tuple(sorted(temporary_array[i: temporary_array_index + 1])))
                            _ = frequent_itemsets[tuple(temporary_array[i: temporary_array_index + 1])]
                        except:
                            # print("except")
                            early_exit = True
                            break

                    if early_exit:
                        input_array_index += 1
                        # print("larger exit, temporary_array:", temporary_array)
                        early_exit = False
                        continue

                    # temporary_array[temporary_array_index] = input_array[input_array_index]
                    generate_combination_helper(input_array, temporary_array, size, combinations,
                                                input_array_index + 1, end, temporary_array_index + 1)
                    input_array_index += 1

        combinations = []  # store all the possible combinations
        temporary_array = [-1] * size  # Temporary array to store current combination
        n = len(input_array)
        generate_combination_helper(input_array, temporary_array, size, combinations,
                                    start=0, end=n - 1, temporary_array_index=0)
        return combinations


    def hash_pair(pair: list, num_buckets = 1000) -> int:
        # pair = sorted(pair)  # baskets are already sorted
        return hash(pair[0] + pair[1]) % num_buckets


    def qualified_as_candidate_pair(pair: list, frequent_itemsets: dict, bitmap: dict) -> bool:
        for item in pair:
            try:
                _ = frequent_itemsets[item]
            except:
                return False
        if bitmap[hash_pair(pair)] == 1:
            return True
        else:
            return False


    def son(baskets: RDD, support_threshold_total=support_threshold_total) -> list:

        def pcy_for_list(partition: list, support_threshold_total=support_threshold_total) -> dict:
            # partition = baskets
            num_baskets_chunk = len(partition)
            support_threshold = math.ceil(support_threshold_total * num_baskets_chunk / num_baskets)

            # first pass
            singleton_counts = {}
            bucket_counts = {}
            for basket in partition:
                for item in basket:
                    singleton_counts[item] = singleton_counts.get(item, 0) + 1

                pairs = generate_combination(basket, size=2)
                for pair in pairs:
                    key = hash_pair(pair)
                    bucket_counts[key] = bucket_counts.get(key, 0) + 1

            for key, value in bucket_counts.items():
                if value >= support_threshold:
                    bucket_counts[key] = 1
                else:
                    bucket_counts[key] = 0

            frequent_itemsets = {}
            for key, value in singleton_counts.items():
                if value >= support_threshold:
                    frequent_itemsets[key] = None  # store all frequent singletons
            # print("singleton_counts", singleton_counts)
            # print("frequent singletons", frequent_itemsets)
            del singleton_counts
            gc.collect()

            # second pass
            itemset_counts = {}
            for basket in partition:
                pairs = generate_combination(basket, size=2)
                for pair in pairs:
                    if qualified_as_candidate_pair(pair, frequent_itemsets, bitmap=bucket_counts):
                        key = tuple(pair)
                        itemset_counts[key] = itemset_counts.get(key, 0) + 1

            for key, value in itemset_counts.items():
                if value >= support_threshold:
                    frequent_itemsets[key] = None  # store all frequent pairs
            # print("pair counts", itemset_counts)
            del itemset_counts
            gc.collect()

            # more passes for larger-size itemsets
            size = 3
            num_frequent_itemsets = len(frequent_itemsets)
            while True:
                itemset_counts = {}
                for basket in partition:
                    itemsets = generate_combination_with_filter(basket, frequent_itemsets, size)
                    for itemset in itemsets:
                        key = tuple(itemset)
                        itemset_counts[key] = itemset_counts.get(key, 0) + 1

                for key, value in itemset_counts.items():
                    if value >= support_threshold:
                        frequent_itemsets[key] = None  # store all frequent pairs
                del itemset_counts
                gc.collect()

                current_num_frequent_itemsets = len(frequent_itemsets)
                # print("frequent_itemsets", frequent_itemsets)
                if current_num_frequent_itemsets == num_frequent_itemsets:  # no more new frequent itemsets
                    # print("break")
                    break

                num_frequent_itemsets = current_num_frequent_itemsets
                size += 1

            # print("frequent_itemsets", frequent_itemsets)
            return frequent_itemsets

        # First stage
        num_baskets = baskets.count()
        candidate_itemsets = dict.fromkeys(baskets.mapPartitions(lambda _: pcy_for_list(list(_), support_threshold_total)).distinct().collect(), 0)
        # print("candidate_itemsets", candidate_itemsets)

        # Second stage
        def qualified_as_candidate_itemset(itemset):
            try:
                _ = candidate_itemsets[itemset]
                return True
            except:
                return False

        singleton_counts = baskets.\
            flatMap(lambda basket: basket).\
            filter(lambda item: qualified_as_candidate_itemset(item)).\
            map(lambda _: (_, 1)).\
            reduceByKey(lambda x,y: x+y).\
            filter(lambda pair: pair[1] >= support_threshold_total).keys().collect()
        frequent_itemsets = [sorted(singleton_counts)]
        del singleton_counts
        gc.collect()

        size = 2
        while True:
            frequent_itemsets_for_particular_size = baskets.\
                flatMap(lambda _: generate_combination_with_filter(_, candidate_itemsets, size)).\
                filter(lambda _: qualified_as_candidate_itemset(tuple(_))).\
                map(lambda _: (tuple(_), 1)).\
                reduceByKey(lambda x,y: x+y).\
                filter(lambda pair: pair[1] >= support_threshold_total).keys().collect()
            if frequent_itemsets_for_particular_size == []:
                break
            else:
                frequent_itemsets.append(sorted(frequent_itemsets_for_particular_size))
                size += 1

            del frequent_itemsets_for_particular_size
            gc.collect()

        return frequent_itemsets


    def pcy_for_rdd(baskets: RDD, support_threshold_total=support_threshold_total) -> list:

        def check_all_subsets_frequent(itemset: list, frequent_itemsets_dict: dict) -> bool:
            '''
            For example, given a triple ['2', '1', '8'], check if all its subsets ['2', '1'], ['2', '8'], ['1', '8']
            are frequent items.
            :param itemset:
            :return:
            '''
            itemset_size = len(itemset)
            for i in range(itemset_size):
                subset = itemset.copy()
                subset.pop(i)
                try:
                    _ = frequent_itemsets_dict[tuple(subset)]  # 不再需要sorted这个subset，basket已sort
                except:
                    return False
            return True

        num_baskets = baskets.count()
        singleton_counts = baskets.\
            flatMap(lambda set: [(item, 1) for item in set]).\
            reduceByKey(lambda x,y: x+y).\
            filter(lambda pair: pair[1] >= support_threshold_total)
        # frequent_singletons_dict = dict(singleton_counts.collect()).keys()
        frequent_itemsets_dict = dict(singleton_counts.collect())
        # print("frequent_itemsets_dict", frequent_itemsets_dict)
        frequent_itemsets_list = [sorted(list(frequent_itemsets_dict.keys()))]
        del singleton_counts
        gc.collect()

        # all_pairs = baskets.flatMap(lambda basket: generate_combination(basket, 2)).persist()  # 既然first/second pass都要用，为何不persist
        #
        # bucket_counts = all_pairs.map(lambda pair:(hash_pair(pair), 1)).reduceByKey(lambda x,y: x+y).collect()  # first pass
        # bitmap = dict(bucket_counts)
        # for key, value in bitmap.items():
        #     if value >= support_threshold_total:
        #         bitmap[key] = 1
        #     else:
        #         bitmap[key] = 0

        current_itemset_size = 2
        while True:
            # print("current_itemset_size", current_itemset_size)
            # if current_itemset_size == 2: # pairs are special
            #     frequent_itemsets = all_pairs.\
            #         filter(lambda _: qualified_as_candidate_pair(_, frequent_itemsets_dict, bitmap)).\
            #         map(lambda pair: (tuple(pair), 1)).\
            #         reduceByKey(lambda x, y: x + y).\
            #         filter(lambda pair: pair[1] >= support_threshold_total).persist()
            #     del all_pairs
            #     gc.collect()
            # else:  # 双重filter
            frequent_itemsets = baskets.flatMap(lambda basket: generate_combination_with_filter(basket, frequent_itemsets_dict, current_itemset_size)). \
                map(lambda itemset: (tuple(itemset), 1)).\
                reduceByKey(lambda x,y: x+y).\
                filter(lambda pair: pair[1] >= support_threshold_total).persist()
            # if frequent_itemsets.count() == 0:
            #     break
            current_size_frequent_itemsets = sorted(frequent_itemsets.keys().collect())
            if current_size_frequent_itemsets == []:
                break

            frequent_itemsets_list.append(current_size_frequent_itemsets)
            frequent_itemsets_dict.update(dict.fromkeys(current_size_frequent_itemsets))
            # frequent_itemsets_dict.update(dict(frequent_itemsets.collect()))
            current_itemset_size += 1
            del frequent_itemsets  # 也许正确操作应该是释放内存之后再del？我不懂
            del current_size_frequent_itemsets
            gc.collect()

        gc.collect()
        return frequent_itemsets_list


    def write_result_to_file(frequent_itemsets: int, output_file_path: str):
        # output_file = open(output_file_path + "/Zhijun_Liao_task1.txt", "w", newline="")
        with open(output_file_path, "w") as output_file:
            # (1) Candidates
            output_file.write("Candidates:")
            output_file.write("\n")

            output_file.write("('{}')".format(frequent_itemsets[0][0]))
            for frequent_singleton in frequent_itemsets[0][1:]:
                output_file.write(",('{}')".format(frequent_singleton))
            output_file.write("\n")
            output_file.write("\n")

            n = len(frequent_itemsets)
            for i in range(1, n):
                itemsets = frequent_itemsets[i]
                output_file.write("{}".format(itemsets[0]))
                for itemset in itemsets[1:]:
                    output_file.write(",{}".format(itemset))
                output_file.write("\n")
                output_file.write("\n")

            # (2) Frequent Itemsets
            output_file.write("Frequent Itemsets:")
            output_file.write("\n")

            output_file.write("('{}')".format(frequent_itemsets[0][0]))
            for frequent_singleton in frequent_itemsets[0][1:]:
                output_file.write(",('{}')".format(frequent_singleton))
            output_file.write("\n")
            output_file.write("\n")

            n = len(frequent_itemsets)
            for i in range(1, n):
                itemsets = frequent_itemsets[i]
                if i == n - 1:
                    output_file.write("{}".format(itemsets[0]))
                    for itemset in itemsets[1:]:
                        output_file.write(",{}".format(itemset))
                else:
                    output_file.write("{}".format(itemsets[0]))
                    for itemset in itemsets[1:]:
                        output_file.write(",{}".format(itemset))
                    output_file.write("\n")
                    output_file.write("\n")


    start_time = time.time()
    baskets = sc.textFile(input_file_path, minPartitions=15). \
        filter(lambda pair: "user_id" not in pair). \
        map(lambda string: string.split(",")). \
        groupByKey(). \
        map(lambda pair: sorted(list(set(pair[1].data)))). \
        filter(lambda _: len(_) > filter_threshold).persist()

    frequent_itemsets = pcy_for_rdd(baskets, support_threshold_total)

    write_result_to_file(frequent_itemsets, output_file_path)
    end_time = time.time()
    run_time = end_time - start_time
    print("Duration:", run_time)


if __name__ == '__main__':
    main(sys.argv)