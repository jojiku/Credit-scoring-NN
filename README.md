# Credit-scoring-NN
Neural networks approach for the credit scoring problem

The core idea of this project is to predit whether a client will default his credit or successfully pay it. According to prediction the decision of providing a credit is given. For that purpose I used data from credit scoring competition based on credit history data from the Alfa-Bank Machine Learning Laboratory and the Open Data Science Community (https://ods.ai/competitions/dl-fintech-bki).

Dataset description:



We will be using a neural approach for two reasons: firstly, because it would allow us to quickly and easily change the output to a simple binary classification problem. Secondly, because the predict_proba functionality allows us to output a probability score (probability of 1), this score is what we will use for predicting the probability of 90 days past due delinquency or worse in 2 years time.
But to evaluate the quality of our neural model, we'll need a baseline. Traditional machine learning methods such as logistic regression, gradient boosting and random forest will be used for that purpose.

Furthermore, we will predominantly be adopting a quantiles based approach in order to streamline the process as much as possible so that hypothetical credit checks can be returned as easily and as quickly as possible.

This model lead to the following metrics on unseen test data:

an accuracy rate of 0.800
accuracy rate of 0.800
accuracy rate of 0.800
accuracy rate of 0.800

I deem this accuracy rate to be acceptable given that we used a relatively simple quantile based approach and in light of the fact that no parameter optimization was undertaken.
