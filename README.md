

This repository covers fundamental techniques in the big data mininng context, including Association Rule Mining (finding frequent itemsets), Recommendation System, Social Network Analysis, Clustering, etc. The content is from USC course *INF 553 Foundations and Applications of Data Mining*. I built all algorithms from scratch with PySpark (version: 2.4.4) and got full scores on all these assignments. I mainly use Spark RDD (Resilient Distributed Datasets), though sometimes would also refer to Spark DataFrame. All data used in the tasks is from the Yelp public review data set [[SOURCE](https://www.yelp.com/dataset)]. 

Below is the summary of each topic and required task.

|  No  |          Topic          |                          Main Task                           |
| :--: | :---------------------: | :----------------------------------------------------------: |
|  1   |        MapReduce        | Getting familiar. Do simple EDA on the dataset, including word count and aggregation (average, max, find top-K). Try diffrenrt repartitioning of the  dataset to compare compuational efficiency. |
|  2   | Association Rule Mining | Find businesses frequently visited together. Use SON algorithm as the main framework, and apply in-memory algorithms like Apriori, PCY, MultiHash to process data in each chunk. |
|  3   |  Recommendation System  | Use Tf-Idf to extract top topic words from review texts to construct user-vectors, business-vectors and build a Content-Based RS. Use Near-Neighbor Search to build a Collaborative Filtering RS. Try improving the system and obtain lower RMSE. |
|  4   | Social Network Analysis | Implement Girvan-Newman algorithm to detect non-overlapping communnities in the customers' network. |



Fight on!
