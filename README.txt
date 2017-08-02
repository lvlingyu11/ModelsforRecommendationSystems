This project aims at learning from the incomplete ratings, which is also the problem as recommendation systems. 

In this project, sereval models/algorithms are implemented as recommendation engines: item based top-N recommendation (tf-idf, cosine cimilarity); EM-based nonnegative matrix factorization; Nearest neighbor (NN). 

The setup of the folder is as:
-RecommendationEngines: This folder includes the implementation of the different recommendation models. 
-Evaluation: It contains the evaluation metrics used for measure the performance of the models.
-DataSets: The folder has part of the data used in the project. The other part of data we used is from the Docear's research paper recommendation paper, which needs permission from the organization. Thus this dataset is not included here. For anyone who is interested in their research paper dataset, please visit their website: 
 	   http://labs.docear.org
-test_nn: The python file to test the nearest neighbor model.
-test_nmf_tfidf: The python file to test the nonnegative matrix factorization and top n recommender models.