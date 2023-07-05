# Credit-scoring-NN
## Neural networks approach for the credit scoring problem

The core idea of this project is to predit whether a client will default his credit or successfully pay it. 
Based on projections, the decision to grant credit is made. For that purpose I used data from credit scoring competition based on credit history data from the Alfa-Bank Machine Learning Laboratory and the Open Data Science Community (https://ods.ai/competitions/dl-fintech-bki).
---
Dataset includes credit history of clients who were previously given with credits.  
We will be using a neural approach as a main method as neural networks in recent years have begun to steadily surpass traditional ml methods in binary classification problems.
But to evaluate the quality of our neural model, we'll need a baseline. Traditional machine learning methods such as logistic regression, gradient boosting and random forest will be used for that purpose.
For convenient use of models the GUI on streamlit was created as well. 

This model lead to the following metrics on unseen test data:

* Accuracy: 0.744	
* Precision: 0.755	
* Recall: 0.722
* F1 score: 0.738
* Roc-auc: 0.743
I deem this accuracy rate to be acceptable given that we used a relatively simple architecture for neural network, which easily surpassed traditional ml approaches.
