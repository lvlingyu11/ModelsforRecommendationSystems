The RecommendationEngines contains the implemented models for recommender systems.

-BaseModel.py: It is the python file for base models for top-n and nonnegative recommendation systems. This model include two abstrat methods: train, which is training the model, and predict, which is used to predict the user ratings

-tf_idf.py: This is the python file which implemented top-n recommender system. Docear research paper dataset is used here. The model works in the following way:
	    First, the research papers are processed and represented as tf-idf vectors. Then consine similarity scores are computed among the papers. Finally, for each document, the top n similar papers are used as recommendation papers for this document.

-NMF.py: In this python file, Expectation Maximization (EM)-based Nonnegative Matrix Factorization (NMF) recommendation model is implemented. The details of the model is described in the paper:
	 Zhang, Sheng, et al. "Learning from incomplete ratings using non-negative matrix factorization." Proceedings of the 2006 SIAM International Conference on Data Mining. Society for Industrial and Applied Mathematics, 2006.
	 http://epubs.siam.org/doi/abs/10.1137/1.9781611972764.58
	 The MovieLens dataset is used here.

-NN.py: This python file implement the user based nearest neighbor recommdation model. The implementaion referred the paper:
	Desrosiers, Christian, and George Karypis. "A comprehensive survey of neighborhood-based recommendation methods." Recommender systems handbook (2011): 107-144.
	http://ai2-s2-pdfs.s3.amazonaws.com/802f/3f316f87c6bc675cc55a2a1bf4bb0f12dd1e.pdf

	First, the consine similarity scores are computed between the users, and then k nearest neighbors are found based on the simlarity scores.
	The missing rating for current user is then computed as the weighted sum of the k nearest neighbors' ratings. And each weight is the corresponding cosine similarity score.
	We also use the MovieLens dataset here.

-Exceptions.py: This python file includes some user defined exceptions.