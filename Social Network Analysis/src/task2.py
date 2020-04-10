from pyspark import SparkContext
import sys
import time

SparkContext.setSystemProperty('spark.executor.memory', '7g')
SparkContext.setSystemProperty('spark.driver.memory', '7g')
sc = SparkContext(appName="INF553HW4", master="local[*]")
sc.setLogLevel("WARN")
sc.setLogLevel("ERROR")

def main(argv):
    filter_threshold = int(argv[1])
    input_file_path = argv[2]
    betweenness_output_file_path = argv[3]
    community_output_file_path = argv[4]

    # filter_threshold = 7
    # input_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW4/data/ub_sample_data.csv"
    # betweenness_output_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW4/output/task2-betweenness.txt"
    # community_output_file_path = "/Users/zhijunliao/Marks/USC/INF-553/HW/INF553HW4/output/task2-final-output.txt"

    def bfs(root, direct_connected_dict: dict) -> list:
        """
        :param root: starting node of the BFS tree
        :param direct_connected_dict: key is a node, value is a set consisting of all nodes directly connected to it
        :return: tree, a list consisting of dicts, each dict corresponds to each level of the BFS tree.
        In each dict, the key-value pair is (node, (parents of this node, number of shortest paths))
        """
        reached, next_level_reached, children = {root}, {root}, {root}
        tree = []
        level = 0

        while next_level_reached:
            next_level_reached = set()
            tree.append({})  # for the new level
            for current_node in children:
                neighbors = direct_connected_dict[current_node].difference(reached)
                # print("current_node", current_node, " / ", "neighbors", neighbors)
                for neighbor in neighbors:
                    # print("neighbor", neighbor)
                    next_level_reached.add(neighbor)
                    try:
                        tree[level][neighbor].add(current_node)
                    except KeyError:
                        tree[level][neighbor] = {current_node}

            reached = reached.union(next_level_reached)
            children = next_level_reached
            level += 1
        tree.pop()

        # There's only one shortest path for nodes at level one of the tree:
        for child, parents in tree[0].items(): tree[0][child] = (parents, 1)
        # For nodes at level two and above, assign the number of shortest paths to them:
        for level in range(1, len(tree)):
            for child, parents in tree[level].items():
                tree[level][child] = (parents, sum([tree[level-1][parent][1] for parent in parents]))
        return tree

    def calculate_betweenness(tree: list, nodes_one_pairs: dict) -> list:
        betweenness_score = []
        credit_assignment = nodes_one_pairs.copy()
        num_shortest_paths_dict = nodes_one_pairs.copy()
        for level in range(len(tree)):
            for child, (parents, num_shortest_paths) in tree[level].items():
                num_shortest_paths_dict[child] = num_shortest_paths

        for level in reversed(range(len(tree))):
            # print("level", level)
            for child, (parents, num_shortest_paths) in tree[level].items():
                # num_parents = len(parents)
                for parent in parents:
                    added_credit = credit_assignment[child] * num_shortest_paths_dict[parent] / num_shortest_paths
                    betweenness_score.append((tuple(sorted((child, parent))), added_credit))
                    credit_assignment[parent] += added_credit
        return betweenness_score

    def find_nodes_in_same_group(current_node, members: set, direct_connected_dict: dict) -> set:
        neighbors = direct_connected_dict[current_node]
        for neighbor in neighbors:
            if neighbor in members:
                continue
            else:
                members.add(neighbor)
                members = find_nodes_in_same_group(neighbor, members, direct_connected_dict)
        return members

    # 38648 records
    total_start = time.time()
    input_data = sc.textFile(input_file_path). \
        filter(lambda line: "user_id" not in line). \
        map(lambda line: tuple(line.split(","))). \
        groupByKey(). \
        mapValues(set). \
        persist()  # 3374

    edges = input_data. \
        cartesian(input_data). \
        filter(lambda pair: pair[0][0] < pair[1][0]). \
        filter(lambda pair: len(pair[0][1].intersection(pair[1][1])) >= filter_threshold). \
        flatMap(lambda pair: [(pair[0][0], pair[1][0]), (pair[1][0], pair[0][0])]). \
        persist()  # 996 498
    vertices = edges.flatMap(lambda _: _).distinct().persist()  # 222

    direct_connected_dict = sc.broadcast(dict(edges.groupByKey().mapValues(set).collect()))
    adjacency_matrix = dict.fromkeys(edges.collect(), 1)
    degrees = {}
    for node, neighbors in direct_connected_dict.value.items():
        degrees[node] = len(neighbors)

    all_nodes = sc.broadcast(vertices.collect())
    nodes_one_pairs = sc.broadcast(dict.fromkeys(all_nodes.value, 1))
    nodes_negative_one_pairs = sc.broadcast(dict.fromkeys(all_nodes.value, -1))
    num_edges = edges.count()

    # for task2_1 output
    intermediate_result = vertices. \
        flatMap(lambda vertex: calculate_betweenness(tree=bfs(vertex, direct_connected_dict.value),
                                                     nodes_one_pairs=nodes_one_pairs.value)). \
        reduceByKey(lambda x, y: x + y). \
        mapValues(lambda values: values / 2). \
        persist()
    intermediate_result_collected = intermediate_result.collect()
    intermediate_result_collected.sort(key=lambda kv: (-kv[1], kv[0][0]))
    with open(betweenness_output_file_path, "w") as output_file:
        for edge, betweenness in intermediate_result_collected:
            output_file.write(f"{edge}, {betweenness}")
            output_file.write("\n")

    # for task2_2 output
    # Run Girvan-Newman algorithm
    modularity = previous_modularity = 0
    communities = previous_communities = {}
    while modularity >= previous_modularity:
        previous_modularity = modularity
        betweenness_score = vertices. \
            flatMap(lambda vertex: calculate_betweenness(tree=bfs(vertex, direct_connected_dict.value),
                                                         nodes_one_pairs=nodes_one_pairs.value)). \
            reduceByKey(lambda x, y: x + y). \
            mapValues(lambda values: values / 2). \
            persist()
        edge_to_drop = betweenness_score.max(lambda pair: pair[1])[0]

        # cut the edge with highest betweenness
        direct_connected_dict = direct_connected_dict.value
        node_1, node_2 = edge_to_drop[0], edge_to_drop[1]
        direct_connected_dict[node_1].remove(node_2)
        direct_connected_dict[node_2].remove(node_1)
        direct_connected_dict = sc.broadcast(direct_connected_dict)

        # assign every node to its corresponding community
        node_membership = nodes_negative_one_pairs.value.copy()
        community_id = 0
        for node in all_nodes.value:
            if node_membership[node] >= 0:
                continue
            members = find_nodes_in_same_group(node, {node}, direct_connected_dict.value)
            for member in members:
                node_membership[member] = community_id
            community_id += 1

        previous_communities = communities.copy()
        communities = {}
        for node, community_id in node_membership.items():
            communities[community_id] = communities.get(community_id, []) + [node]

        # calculate current modularity
        modularity = 0
        for community_id, members in communities.items():
            for i in range(len(members) - 1):
                for j in range(i + 1, len(members)):
                    Aij = 1 if (members[i], members[j]) in adjacency_matrix else 0
                    modularity += Aij - degrees[members[i]] * degrees[members[j]] / num_edges
        modularity = modularity / num_edges
        print("modularity", modularity)

    # write result to file
    final_output = [sorted(members) for members in previous_communities.values()]
    final_output.sort(key=lambda _: (len(_), _[0]))
    with open(community_output_file_path, "w") as output_file:
        for output in final_output:
            output_file.write(f"'{output[0]}'")
            for user in output[1:]:
                output_file.write(f", '{user}'")
            output_file.write("\n")
    print("total running time:  ", time.time() - total_start)


if __name__ == '__main__':
    main(sys.argv)
