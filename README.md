# Search_Results_Relevance

###1. Text Preprocessing
  lower-case transformation
  stop-words elimination
  stemming
  query extention(\*1)

###2. Feature Engineering
  q_len; word length of query  
  t_len; word length of title  
  d_len; word length of description  

  nleven1; normalized levenshtein distance(shortest alignment) between query and title  
  nleven2; normalized levenshtein distance(longest alignment) between query and title  
  sorensen; sorensen distance between query and title  
  jaccard; jaccard distance between query and title  
  ncd; normalized compression distance between query and title  
  cos_dist_1_100; TF-IDF cosine distance between query and title(ngram=1, svd_component=100)  
  cos_dist_1_250; TF-IDF cosine distance between query and title(ngram=1, svd_component=250)  
  cos_dist_1_500; TF-IDF cosine distance between query and title(ngram=1, svd_component=500)  
  cos_dist_1_1000; TF-IDF cosine distance between query and title(ngram=1, svd_component=1000)  
  cos_dist_2_100; TF-IDF cosine distance between query and title(ngram=2, svd_component=100)  
  cos_dist_2_250; TF-IDF cosine distance between query and title(ngram=2, svd_component=250)  
  cos_dist_2_500; TF-IDF cosine distance between query and title(ngram=2, svd_component=500)  
  cos_dist_2_1000; TF-IDF cosine distance between query and title(ngram=2, svd_component=1000)  
  cos_dist_3_100; TF-IDF cosine distance between query and title(ngram=3, svd_component=100)  
  cos_dist_3_250; TF-IDF cosine distance between query and title(ngram=3, svd_component=250)  
  cos_dist_3_500; TF-IDF cosine distance between query and title(ngram=3, svd_component=500)  
  cos_dist_3_1000; TF-IDF cosine distance between query and title(ngram=3, svd_component=1000)  

  sorensen_ex; sorensen distance between extended query and title  
  jaccard_ex; jaccard distance between extended query and title  
  ncd_ex; normalized compression distance between extended query and title  
  cos_dist_1_100_ex; TF-IDF cosine distance between extended query and title(ngram=1, svd_component=100)  
  cos_dist_1_250_ex; TF-IDF cosine distance between extended query and title(ngram=1, svd_component=250)  
  cos_dist_1_500_ex; TF-IDF cosine distance between extended query and title(ngram=1, svd_component=500)  
  cos_dist_1_1000_ex; TF-IDF cosine distance between extended query and title(ngram=1, svd_component=1000)  
  cos_dist_2_100_ex; TF-IDF cosine distance between extended query and title(ngram=2, svd_component=100)  
  cos_dist_2_250_ex; TF-IDF cosine distance between extended query and title(ngram=2, svd_component=250)  
  cos_dist_2_500_ex; TF-IDF cosine distance between extended query and title(ngram=2, svd_component=500)  
  cos_dist_2_1000_ex; TF-IDF cosine distance between extended query and title(ngram=2, svd_component=1000)  
  cos_dist_3_100_ex; TF-IDF cosine distance between extended query and title(ngram=3, svd_component=100)  
  cos_dist_3_250_ex; TF-IDF cosine distance between extended query and title(ngram=3, svd_component=250)  
  cos_dist_3_500_ex; TF-IDF cosine distance between extended query and title(ngram=3, svd_component=500)  
  cos_dist_3_1000_ex; TF-IDF cosine distance between extended query and title(ngram=3, svd_component=1000)  

  qid; query id(\*2)  
  max_relevance; max relevance for the query(\*2)  
  min_relevance; min relevance for the query  
  mean_relevance; mean relevance for the query  

  \*1) extended query = query added title texts whose relevance = 4 and variance = 0.0.  
  \*2) note that train/test data contain lot of common queries  

###3. Models
  svm regressor  
  random forest regressor  
  gradient boosting regressor  

  hyperparameter tuning is based on cross validation with mean square error.
  
  best scores(MSE) for each model are below:  
  	svm regressor: 0.5265  
	random forest regressor: 0.4686   
	gradient boosting regressor: 0.4665  
  	
###4. Stacking
  score = svr + α \* rfr + β \* gbr  
  where α,β are tuned from [2^-10, ..., 2^-1, 2^0, 2^1, ..., 2^10] by cross validation with kappa estimator.
  
  transformation from continuous values to labels is done as follows:  
  1. sort scores by the descending order  
  2. select 4,3,2,1 as same ratio as training data  

###5. Result  
  private score = 0.66830 (If submitted during the competition, it would have been 183th place)
  
  
