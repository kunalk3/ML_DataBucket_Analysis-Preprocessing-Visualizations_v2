<div align="right">
<img src="https://user-images.githubusercontent.com/41562231/141720820-090897f9-f564-45e2-9265-15c1269db795.png" height="120" width="900">
</div>

# __Popular Machine Learning Metrics.__

__Introduction-__
Choosing the right metric is crucial while evaluating machine learning (ML) models. Various metrics are proposed to evaluate ML models in different applications, and I thought it may be helpful to provide a summary of popular metrics in a here, for better understanding of each metric and the applications they can be used for. In some applications looking at a single metric may not give you the whole picture of the problem you are solving, and you may want to use a subset of the metrics discussed in this post to have a concrete evaluation of your models.

- __Classification Metrics__ (accuracy, precision, recall, F1-score, ROC, AUC, log-loss)
- __Regression Metrics__ (MSE, MAE, RMSE, Adj R². R²)
- __Ranking Metrics__ (MRR, DCG, NDCG)
- __Statistical Metrics__ (Correlation)
- __Computer Vision Metrics__ (PSNR, SSIM, IoU)
- __NLP Metrics__ (Perplexity, BLEU score)
- __Deep Learning Related Metrics__ (Inception score, Frechet Inception distance)

---

## 🧲 Part 1: Classification & Regression Evaluation Metrics
```diff
- A metric is different from loss function?

    Loss functions are functions that show a measure of the model performance and are used to train a machine learning 
    model (using some kind of optimization), and are usually differentiable in model’s parameters. On the other hand, 
    metrics are used to monitor and measure the performance of a model (during training, and test), and do not need to 
    be differentiable. However if for some tasks the performance metric is differentiable, it can be used both as a loss 
    function (perhaps with some regularizations added to it), and a metric, such as MSE.
```
<div align="center">
    <h2><b>🦮 Classification Related Metrics</b></h2>
    <i>Models such as support vector machine (SVM), logistic regression, decision trees, random forest, XGboost, convolutional neural network¹, recurrent neural network are some of the most popular classification models.</i>
</div><br>

---

__📌 Confusion Matrix:__

One of the key concept in classification performance is confusion matrix (AKA error matrix), which is a tabular visualization of the model predictions versus the ground-truth labels. Each row of confusion matrix represents the instances in a predicted class and each column represents the instances in an actual class.

<div align="center">
    <img src="Assets/1.jpg" height="100" width="400">
</div>

- __Out of 100 cat images__ the model has predicted 90 of them correctly and has mis-classified 10 of them. If we refer to the “cat” class as positive and the non-cat class as negative class, then 90 samples predicted as cat are considered as as __true-positive__, and the 10 samples predicted as non-cat are __false negative__.
- __Out of 1000 non-cat images__, the model has classified 940 of them correctly, and mis-classified 60 of them. The 940 correctly classified samples are referred as __true-negative__, and those 60 are referred as __false-positive__.

_Another Example-_

<div align="center">
    <img src="Assets/10.jpg" height="250" width="500">
</div>

- _True Positive:_ We predicted positive and it’s true. In the image, we predicted that a woman is pregnant and she actually is.
- _True Negative:_ We predicted negative and it’s true. In the image, we predicted that a man is not pregnant and he actually is not.
- _False Positive (Type 1 Error)_- We predicted positive and it’s false. In the image, we predicted that a man is pregnant but he actually is not.
- _False Negative (Type 2 Error)_- We predicted negative and it’s false. In the image, we predicted that a woman is not pregnant but she actually is.

```python
from sklearn.metrics import confusion_matrix
y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
```

    Output:
    array([ [2, 0, 0],
            [0, 0, 1],
            [1, 0, 2] ])

---

__📌 Classification Accuracy:__

Classification accuracy is perhaps the simplest metrics one can imagine, and is defined as __the number of correct predictions divided by the total number of predictions__, multiplied by 100. So in the above example, out of 1100 samples 1030 are predicted correctly, resulting in a classification accuracy of:

<div align="center">
    <img src="Assets/9.jpg" height="60" width="300">
</div>

    Classification accuracy = (90+940)/(1000+100)
                            = 1030/1100
                            = 93.6%

```python
from sklearn.metrics import confusion_matrix, accuracy_score
threshold=0.5
preds_list = preds_list >= threshold
tn, fp, fn, tp = confusion_matrix(labels_list, preds_list).ravel()
accuracy = accuracy_score(labels_list, preds_list
```

---

__📌 Precision:__

There are many cases in which classification accuracy is not a good indicator of your model performance. One of these scenarios is when your class distribution is imbalanced (one class is more frequent than others). In this case, even if you predict all samples as the most frequent class you would get a high accuracy rate, which does not make sense at all (because your model is not learning anything, and is just predicting everything as the top class). For example in our cat vs non-cat classification above, if the model predicts all samples as non-cat, it would result in a 1000/1100= 90.9%. 

__Precision = True_Positive/ (True_Positive+ False_Positive)__

<div align="center">
    <img src="Assets/11.jpg" height="60" width="400">
</div>

    Precision_cat = samples correctly predicted cat/samples predicted as cat 
                  = 90/(90+60) 
                  = 60%
    Precision_NonCat = 940/950
                     = 98.9%

```python
from sklearn.metrics import precision_score
y_true = [0, 1, 1, 0, 1, 0]
y_pred = [0, 0, 1, 0, 0, 1]
precision_score(y_true, y_pred)
```
    output:
    0.5

---

__📌 Recall:__

Recall is another important metric, which is defined as the fraction of samples from a class which are correctly predicted by the model.

__Recall = True_Positive/ (True_Positive+ False_Negative)__
    
<div align="center">
    <img src="Assets/12.jpg" height="60" width="400">
</div>

    Recall_cat = 90/100
               = 90%
    Recall_NonCat = 940/1000
                  = 94%

```python
from sklearn.metrics import recall_score
y_true = [0, 1, 1, 0, 1, 0]
y_pred = [0, 0, 1, 0, 0, 1]
recall_score(y_true, y_pred)
```
    output:
    0.333333

---

__📌 F1 Score:__

Depending on application, you may want to give higher priority to recall or precision. But there are many applications in which both recall and precision are important. Therefore, it is natural to think of a way to combine these two into a single metric. One popular metric which __combines precision and recall is called F1-score__, which is the harmonic mean of precision and recall

__F1-score= 2*Precision*Recall/(Precision+Recall)__

<div align="center">
    <img src="Assets/13.jpg" height="60" width="400">
</div>

    F1_cat = 2*0.6*0.9/(0.6+0.9)
           = 72%

```python
from sklearn.metrics import f1_score
y_true = [[0, 0, 0], [1, 1, 1], [0, 1, 1]]
y_pred = [[0, 0, 0], [1, 1, 1], [1, 1, 0]]
f1_score(y_true, y_pred, average=None)
```
    output:
    array([0.66666667, 1. , 0.66666667])

__Precision/Recall Trade-off:__ If you want to make the precision too high, you would end up seeing a drop in the recall rate, and vice versa. _This is called Precision/Recall Trade-off._

```python
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_true, y_predicted)
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.show()
```

<div align="center">
    <img src="Assets/27.jpg" height="200" width="500">
</div>

_Note: As you can see as the threshold increases precision increases but at the cost of recall. From this graph, one can pick a suitable threshold as per their requirements._

---

__📌 Sensitivity and Specificity:__
Sensitivity and specificity are two other popular metrics mostly used in medical and biology related fields which are very sensitive to the data.

__Sensitivity = Recall = TP/(TP+FN)__

__Specificity = True Negative Rate = TN/(TN+FP)__

---

__📌 ROC Curve:__
The receiver operating characteristic curve is plot which shows the performance of a binary classifier as function of its cut-off threshold. It essentially shows the __true positive rate (TPR) against the false positive rate (FPR)__ for various threshold values.

- Many of the classification models are probabilistic, i.e. they predict the probability of a sample being a cat. They then compare that output probability with some cut-off threshold and if it is larger than the threshold they predict its label as cat, otherwise as non-cat. 
- As an example your model may predict the below probabilities for __4 sample images: [0.45, 0.6, 0.7, 0.3]__. Then depending on the __threshold values__ below, you will get different labels:

        cut-off= 0.5: predicted-labels= [0,1,1,0] (default threshold)
        cut-off= 0.2: predicted-labels= [1,1,1,1]
        cut-off= 0.8: predicted-labels= [0,0,0,0]

        ROC curve essentially finds out the TPR and FPR for various threshold values and plots TPR against the FPR

<div align="center">
    <img src="Assets/2.jpg" height="300" width="400">
</div>

- From figure, the lower the cut-off threshold on positive class, the more samples predicted as positive class, i.e. higher true positive rate (recall) and also higher false positive rate (corresponding to the right side of this curve). Therefore, there is a trade-off between how high the recall could be versus how much we want to bound the error (FPR).

```python
from sklearn.metrics import roc_auc_score 
roc_auc = roc_auc_score(labels, predictions)
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.show()
```

---

__📌 AUC:__

The __Area Under the Curve (AUC)__, is an aggregated measure of performance of a binary classifier on all possible threshold values (and therefore it is threshold invariant). AUC calculates the area under the ROC curve, and therefore it is between 0 and 1. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example

<div align="center">
    <img src="Assets/3.jpg" height="250" width="400">
</div>

- On high-level, the higher the AUC of a model the better it is. But sometimes threshold independent measure is not what you want, e.g. you may care about your model recall and require that to be higher than 99% (while it has a reasonable precision or FPR). In that case, you may want to tune your model threshold such that it meets your minimum requirement on those metrics (and you may not care even if you model AUC is not too high).
- Therefore in order to decide how to evaluate your classification model performance, perhaps you want to have a good understanding of the business/problem requirement and the impact of low recall vs. low precision, and decide what metric to optimize for.
- From a practical standpoint, a classification model which outputs probabilities is preferred over a single label output, as it provides the flexibility of tuning the threshold such that it meets your minimum recall/precision requirements. Not all models provide this nice probabilistic outputs though, e.g. SVM does not provide a simple probability as an output (although it provides margin which can be used to tune the decision, but it is not as straightforward and interpretable as having output probabilities).

```python
import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
metrics.auc(fpr, tpr)
```

---

__📌 Log Loss:__

Log loss (Logistic loss) or Cross-Entropy Loss is one of the major metrics to assess the performance of a classification problem.

- There is an minor issue with AUC ROC, it only takes into account the order of probabilities and hence it does not take into account the model’s capability to predict higher probability for samples more likely to be positive. In that case, we could us the log loss which is nothing but negative average of the log of corrected predicted probabilities for each instance.
- So, lower the log loss, better the model. However, there is no absolute measure on a good log loss and it is use-case/application dependent.

<div align="center">
    <img src="Assets/24.jpg" height="60" width="300">
</div>

    p(yi)   = predicted probability of positive class
    1-p(yi) = predicted probability of negative class
    yi      = 1 for positive class and 0 for negative class (actual values)

```python
from sklearn.metrics import log_loss
log_loss(["spam", "ham", "ham", "spam"], [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
```

---

__📌 Kolomogorov Smirnov chart:__

K-S or Kolmogorov-Smirnov chart measures performance of classification models. More accurately, K-S is a measure of the degree of separation between the positive and negative distributions. The K-S is 100, if the scores partition the population into two separate groups in which one group contains all the positives and the other all the negatives.

- We can also plot the %Cumulative Good and Bad to see the maximum separation. The metrics covered till here are mostly used in classification problems. Till here, we learnt about confusion matrix, lift and gain chart and kolmogorov-smirnov chart.

<div align="center">
    <img src="Assets/25.jpg" height="350" width="500">
</div>

```python
import pandas as pd
from scipy.stats import ks_2samp

df = pd.read_csv('DummyData.csv') # Dummy example
data1 = df.iloc[:,0]
data2 = df.iloc[:,1]
test = ks_2samp(data1,data2)
print(test)
```

    Output:
    Ks_2sampResult(statistic=0.16666666666666663, pvalue=0.7600465102607566)

---

__📌 Gini Coefficient:__

Gini coefficient is sometimes used in classification problems. Gini coefficient can be straigh away derived from the AUC ROC number. Gini is nothing but ratio between area between the ROC curve and the diagnol line & the area of the above triangle.
    
    Gini = (2 * AUC) – 1

- Gini above 60% is a good model.

```python
from typing import List
from itertools import combinations
import numpy as np

def gini(x: List[float]) -> float:
    x = np.array(x, dtype=np.float32)
    n = len(x)
    diffs = sum(abs(i - j) for i, j in combinations(x, r=2))
    return diffs / (2 * n**2 * x.mean())
```

---

__📌 Concordant – Discordant ratio:__

Use case: we have 3 students who have some likelihood to pass this year.

    Following are our predictions :
    A – 0.9
    B – 0.5
    C – 0.3

    if we were to fetch pairs of two from these three student then,
    AB , BC, CA

The concordant pair is where the probability of responder was higher than non-responder. Whereas discordant pair is where the vice-versa holds true.
    
    AB  – Concordant
    BC – Discordant

_Note: Concordant ratio of more than 60% is considered to be a good model. This metric generally is not used when deciding how many customer to target etc. It is primarily used to access the model’s predictive power. For decisions like how many to target are again taken by KS / Lift charts._

---

<div align="center">
    <h2><b>📈 Regression Related Metrics</b></h2>
    <i>Regression models are another family of machine learning and statistical models, which are used to predict a continuous target values.</i>
</div><br>

---

__📌 MSE:__

__Mean squared Error__ is perhaps the most popular metric used for regression problems. It essentially finds the average squared error between the predicted and actual values. It represents the squared distance between actual and predicted values. we perform squared to avoid the cancellation of negative terms and it is the benefit of MSE.

<div align="center">
    <img src="Assets/4.jpg" height="50" width="200">
</div>

```python
from sklearn.metrics import mean_squared_error
print("MSE",mean_squared_error(y_test,y_pred))
```

- Advantages of MSE:
    - The graph of MSE is differentiable, so you can easily use it as a loss function.

- Disadvantages of MSE:
    - The value you get after calculating MSE is a squared unit of output. for example, the output variable is in meter(m) then after calculating MSE the output we get is in meter squared.
    - If you have outliers in the dataset then it penalizes the outliers most and the calculated MSE is bigger. So, in short, It is not Robust to outliers which were an advantage in MAE.

---

__📌 MAE:__

__Mean Absolute Error__ _(mean absolute deviation)_ is another metric which finds the average absolute distance between the predicted and target values. __MAE is known to be more robust to the outliers than MSE.__ The main reason being that in MSE by squaring the errors, the outliers (which usually have higher errors than other samples) get more attention and dominance in the final error and impacting the model parameters.

find the difference between the actual value and predicted value that is an absolute error but we have to find the mean absolute of the complete dataset. Hence sum all the errors and divide them by a total number of observations And this is MAE. And we aim to get a minimum MAE because this is a loss.

<div align="center">
    <img src="Assets/5.jpg" height="150" width="300">
</div>

```python
from sklearn.metrics import mean_absolute_error
print("MAE",mean_absolute_error(y_test,y_pred))
```

- Advantages of MAE:
    - The MAE you get is in the same unit as the output variable.
It is most Robust to outliers.
- Disadvantages of MAE:
    - The graph of MAE is not differentiable so we have to apply various optimizers like Gradient descent which can be differentiable.

---

__📌 RMSE:__

__Root Mean Squared Error__ is clear by the name itself, that it is a simple square root of mean squared error.

<div align="center">
    <img src="Assets/6.jpg" height="60" width="200">
</div>

```python
from sklearn.metrics import mean_squared_error
print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))
```

- Advantages of RMSE:
    - The output value you get is in the same unit as the required output variable which makes interpretation of loss easy.
- Disadvantages of RMSE:
    - It is not that robust to outliers as compared to MAE.
    - RMSE is highly affected by outlier values. Hence, make sure you’ve removed outliers from your data set prior to using this metric.

---

__📌 RMSLE:__

__Root Mean Squared Log Error__ Taking the log of the RMSE metric slows down the scale of error. _The metric is very helpful when you are developing a model without calling the inputs._ In that case, the output will vary on a large scale.

<div align="center">
    <img src="Assets/26.jpg" height="150" width="400">
</div>

```python
from sklearn.metrics import mean_squared_log_error
mean_squared_log_error(y_true, y_pred)
```
- RMSLE is usually used when we don’t want to penalize huge differences in the predicted and the actual values when both predicted and true values are huge numbers.

        - If both predicted and actual values are small: RMSE and RMSLE are same.
        - If either predicted or the actual value is big: RMSE > RMSLE
        - If both predicted and actual values are big: RMSE > RMSLE (RMSLE becomes almost negligible)
---

__📌 R Squared (R²):__ _(coefficient of determination)_

R2 score is a metric that tells the performance of your model, not the loss in an absolute sense that how many wells did your model perform.

<div align="center">
    <img src="Assets/7.jpg" height="100" width="300">
</div>

```python
from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_pred)
print(r2)
```

- If the R2 score is zero then the above regression line by mean line is equal means 1 so 1-1 is zero. So, in this case, both lines are overlapping means model performance is worst, It is not capable to take advantage of the output column.
- when the R2 score is 1, it means when the division term is zero and it will happen when the regression line does not make any mistake, it is perfect. In the real world, it is not possible.

---

__📌 Adjusted R Squared:__

The disadvantage of the R2 score is while adding new features in data the R2 score starts increasing or remains constant but it never decreases because It assumes that while adding more data variance of data increases. But the problem is when we add an irrelevant feature in the dataset then at that time R2 sometimes starts increasing which is incorrect. Hence, To control this situation Adjusted R Squared came into existence.

<div align="center">
    <img src="Assets/8.jpg" height="150" width="300">
</div>

```python
n = 40
k = 2
adj_r2_score = 1 - ((1-r2)*(n-1)/(n-k-1))
print(adj_r2_score)
```

---

## 🧲 Part 2: Ranking, & Statistical Evaluation Metrics

<div align="center">
    <h2><b>📚 Ranking Related Metrics</b></h2>
    <i>Ranking is a fundamental problem in machine learning, which tries to rank a list of items based on their relevance in a particular task (e.g. ranking pages on Google based on their relevance to a given query). </i>
    
    It has a wide range of applications in E-commerce, and search engines:

    - Movie recommendation (as in Netflix, and YouTube),
    - Page ranking on Google,
    - Ranking E-commerce products on Amazon,
    - Query auto-completion,
    - Image search on vimeo,
    - Hotel search on Expedia/Booking.
</div><br>

The algorithms for ranking problem can be grouped into:
- __i) Point-wise models:__ which try to predict a (matching) score for each query-document pair in the dataset, and use it for ranking the items.
- __ii) Pair-wise models:__ which try to learn a binary classifier that can tell which document is more relevant to a query, given pair of documents.
- __iii) List-wise models:__ which try to directly optimize the value of one of the above evaluation measures, averaged over all queries in the training data.

There are various metrics proposed for evaluating ranking problems, such as,

    MRR
    Precision@ K
    DCG & NDCG
    MAP
    Kendall’s tau
    Spearman’s rho

---

__📌 MRR:__

Mean reciprocal rank (MRR) is one of the simplest metrics for evaluating ranking models. MRR is essentially the average of the reciprocal ranks of “the first relevant item” for a set of queries Q

<div align="center">
    <img src="Assets/14.jpg" height="60" width="200">
</div>

let’s consider the below example, in which the model is trying to predict the plural form of English words by masking 3 guess. In each case, the correct answer is also given.

<div align="center">
    <img src="Assets/15.jpg" height="100" width="600">
</div>

---

__The MRR of this system can be found as:__
    
    MRR = 1/3 * (1/2+1/3+1/1)
        = 11/18

    Note: One of the limitations of MRR is that, it only takes the rank of one of the items (the most relevant one) into account, and ignores other items (for example mediums as the plural form of medium is ignored). This may not be a good metric for cases that we want to browse a list of related items.

```python
import pandas as pd

gts = pd.DataFrame.from_dict([ {'query': 'q1', 'document': 'doc2'},
                               {'query': 'q1', 'document': 'doc3'},
                               {'query': 'q2', 'document': 'doc7'},])

results = pd.DataFrame.from_dict([ {'query': 'q1', 'document': 'doc1', 'rank': 1},
                                   {'query': 'q1', 'document': 'doc2', 'rank': 2},
                                   {'query': 'q1', 'document': 'doc3', 'rank': 3},
                                   {'query': 'q2', 'document': 'doc4', 'rank': 1},
                                   {'query': 'q2', 'document': 'doc5', 'rank': 2},
                                   {'query': 'q2', 'document': 'doc6', 'rank': 3},])

MAX_RANK = 10000
hits = pd.merge(gts, results, on=["query", "document"], how="left").fillna(MAX_RANK)

MRR = (1 / hits.groupby('query')['rank'].min()).mean()

'''
### -------  Using sklearn library ------- ###
import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
y_true = np.array([[1, 0, 0], [0, 0, 1]])
y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
label_ranking_average_precision_score(y_true, y_score)
'''
```

---

__📌 Precision at k:__

The number of relevant documents among the top k documents

<div align="center">
    <img src="Assets/16.jpg" height="60" width="400">
</div>

- As an example, if you search for “hand sanitizer” on Google, and in the first page, 8 out of 10 links are relevant to hand sanitizer, then the P@10 for this query equals to 0.8.
Now to find the precision at k for a set of queries Q, you can find the average value of P@k for all queries in Q.
- P@k has several limitations. Most importantly, it fails to take into account the positions of the relevant documents among the top k. Also it is easy to evaluate the model manually in this case, since only the top k results need to be examined to determine if they are relevant or not.

```python
import numpy as np
from sklearn.metrics import top_k_accuracy_score

y_true = np.array([0, 1, 2, 2])
y_score = np.array([[0.5, 0.2, 0.2],  # 0 is in top 2
                     [0.3, 0.4, 0.2],  # 1 is in top 2
                     [0.2, 0.4, 0.3],  # 2 is in top 2
                     [0.7, 0.2, 0.1]]) # 2 isn't in top 2

print(top_k_accuracy_score(y_true, y_score, k=2))
```
    output: 
    0.75

---

__📌 DCG and NDCG:__

Normalized Discounted Cumulative Gain (NDCG) is perhaps the most popular metric for evaluating learning to rank systems. In contrast to the previous metrics, NDCG takes the order and relative importance of the documents into account, and values putting highly relevant documents high up the recommended lists.

Before giving the official definition NDCG, let’s first introduce two relevant metrics, Cumulative Gain (CG) and Discounted Cumulative Gain (DCG).
<div align="center">
    <img src="Assets/17.jpg" height="500" width="800">
</div>

__Normalized Discounted Cumulative Gain (NDCG)__ tries to further enhance DCG to better suit real world applications. Since the retrieved set of items may vary in size among different queries or systems, NDCG tries to compare the performance using the normalized version of DCG (by dividing it by DCG of the ideal system). In other words, it sorts documents of a result list by relevance, finds the highest DCG (achieved by an ideal system) at position p, and used to normalize DCG

_Note: One of its main limitations is that it does not penalize for bad documents in the result. It may not be suitable to measure performance of queries that may often have several equally good results._

```python
from sklearn.metrics import ndcg_score, dcg_score
import numpy as np
  
true_relevance = np.asarray([[3, 2, 1, 0, 0]])   # Relevance scores in Ideal order
relevance_score = np.asarray([[3, 2, 0, 0, 1]])  # Relevance scores in output order

dcg = dcg_score(true_relevance, relevance_score) # DCG score  
idcg = dcg_score(true_relevance, true_relevance) # IDCG score
ndcg = dcg / idcg                                # Normalized DCG score
  
# Using scikit-learn ndcg_score
print("nDCG score (from function) : ", ndcg_score(true_relevance, relevance_score))
```

---

<div align="center">
    <h2><b>⚔️ Statistical Metrics</b></h2>
    Some of the popular metrics here include: Pearson correlation coefficient, coefficient of determination (R²), Spearman’s rank correlation coefficient, p-value, and more². Here we briefly introduce correlation coefficient, and R-squared
</div><br>

---

__📌Pearson Correlation Coefficient:__

Pearson correlation coefficient is perhaps one of the most popular metrics in the whole statistics and machine learning area. Its application is so broad that is used in almost every aspects of statistical modeling, from feature selection and dimensionality reduction, to regularization and model evaluation.

Correlation coefficient of two random variables (or any two vector/matrix) shows their statistical dependence.

- The correlation coefficient of two variables is always a __value in [-1,1].__ _Two variables are known to be independent if and only if their __correlation is 0.___

```python
import pandas as pd
from scipy.stats import pearsonr
 
df = pd.read_csv("Sample_data.csv")
list1 = df['variable_1']
list2 = df['variable_2']
 
corr, _ = pearsonr(list1, list2)
print('Pearsons correlation: %.3f' % corr)
```

---

## 🧲 Part 3: Computer Vision Evaluation Metrics

<div align="center">
    <h2><b>🧑‍🦳 Computer Vision Metrics</b></h2>
    <i>More recently, with the popularization of the convolutional neural networks (CNN) and GPU-accelerated deep-learning frameworks, object- detection algorithms started being developed from a new perspective. CNNs such as R-CNN, Fast R-CNN, Faster R-CNN, R-FCN, SSD and Yolo have highly increased the performance standards on the field.</i>

    Object detection metrics serve as a measure to assess how well the model performs on an object detection task. It also enables us to compare multiple detection systems objectively or compare them to a benchmark. In most competitions, the average precision (AP) and its derivations are the metrics adopted to assess the detections and thus rank the teams.
</div><br>

---

__📌 IoU:__

Guiding principle in all state-of-the-art metrics is the so-called __Intersection-over-Union (IoU) overlap measure__. It is quite literally defined as the intersection over union of the detection bounding box and the ground truth bounding box.

Dividing the area of overlap between predicted bounding box and ground truth by the area of their union yields the Intersection over Union.

__An Intersection over Union score > 0.5 is normally considered a good prediction.__

<div align="center">
    <img src="Assets/18.jpg" height="250" width="500">
</div>

IoU metric determines how many objects were detected correctly and how many false positives were generated (will be discussed below).

- _True Positives [TP]-_ 
Number of detections with __IoU > 0.5__

- _False Positives [FP]-_ Number of detections with __IoU <= 0.5__ or detected more than once

- _False Negatives [FN]-_ Number of objects that not detected or detected with __IoU <= 0.5__

- _Precision-_ Precision measures how accurate your predictions are. i.e. the percentage of your predictions that are correct.

    Precision = True positive / (True positive + False positive)

- _Recall-_ Recall measures how good you find all the positives. 

    Recall = True positive / (True positive + False negative)

- _F1 Score-_ F1 score is HM (Harmonic Mean) of precision and recall.

<div align="center">
    <img src="Assets/19.jpg" height="150" width="600">
</div>

__i) AP -__ The general definition for the Average Precision(AP) is finding the area under the precision-recall curve.

__ii) mAP -__ The mAP for object detection is the average of the AP calculated for all the classes. mAP@0.5 means that it is the mAP calculated at IOU threshold 0.5.

<div align="center">
    <img src="Assets/20.jpg" height="250" width="350">
</div>

__iii) mAP Vs other metric -__ The mAP is a good measure of the sensitivity of the neural network. So good mAP indicates a model that's stable and consistent across different confidence thresholds. Precision, Recall and F1 score are computed for given confidence threshold.

- Which metric is more important ?

In general to analyse better performing models, it's advisable to use both validation set (data set that is used to tune hyper-parameters) and test set (data set that is used to assess the performance of a fully-trained model).

    a) On validation set-

    Use mAP to select the best performing model (model that is more stable and consistent) out of all the trained weights across iterations/epochs. Use mAP to understand whether the model should be trained/tuned further or not.

    Check class level AP values to ensure the model is stable and good across the classes. As per use-case/application, if you're completely tolerant to FNs and highly intolerant to FPs then to train/tune the model accordingly use Precision. As per use-case/application, if you're completely tolerant to FPs and highly intolerant to FNs then to train/tune the model accordingly use Recall.

    b) On test set-

    If you're neutral towards FPs and FNs, then use F1 score to evaluate the best performing model.
    
    If FPs are not acceptable to you (without caring much about FNs) then pick the model with higher Precision. If FNs are not acceptable to you (without caring much about FPs) then pick the model with higher Recall

    Once you decide metric you should be using, try out multiple confidence thresholds (say for example - 0.25, 0.35 and 0.5) for given model to understand for which confidence threshold value the metric you selected works in your favour and also to understand acceptable trade off ranges (say you want Precision of at least 80% and some decent Recall). Once confidence threshold is decided, you use it across different models to find out the best performing model.

<div align="center">
    <img src="Assets/28.jpg" height="150" width="450">
</div>

```python
from collections import namedtuple
import numpy as np
import cv2

Detection = namedtuple("Detection", ["image_path", "gt", "pred"]) # define the `Detection` object

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1) # compute the area of intersection rectangle
	
    # compute the area of both the prediction and ground-truth
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	
    # compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	return iou

examples = [
	Detection("image_0002.jpg", [39, 63, 203, 112], [54, 66, 198, 114]),
	Detection("image_0016.jpg", [49, 75, 203, 125], [42, 78, 186, 126]),
	Detection("image_0075.jpg", [31, 69, 201, 125], [18, 63, 235, 135]),
	Detection("image_0090.jpg", [50, 72, 197, 121], [54, 72, 198, 120]),
	Detection("image_0120.jpg", [35, 51, 196, 110], [36, 60, 180, 108])]

for detection in examples:
	image = cv2.imread(detection.image_path) # load the image

	cv2.rectangle(image, tuple(detection.gt[:2]), tuple(detection.gt[2:]), (0, 255, 0), 2)
	cv2.rectangle(image, tuple(detection.pred[:2]), tuple(detection.pred[2:]), (0, 0, 255), 2)
	
	iou = bb_intersection_over_union(detection.gt, detection.pred) # compute the intersection over union and display it
	cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
	print("{}: {:.4f}".format(detection.image_path, iou))
	cv2.imshow("Image", image) # show the output image
	cv2.waitKey(0)
```

---

__📌 PSNR:__

__Peak signal-to-noise ratio (PSNR)__ is the ratio between the maximum possible power of an image and the power of corrupting noise that affects the quality of its representation. If we have 8-bit pixels, then the values of the pixel channels must be from 0 to 255. By the way, the red, green, blue or RGB color model fits best for the PSNR. PSNR shows a ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation.

    Input image, specified as scalar, vector, or matrix.
    
    Data Types: single | double | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | Boolean | fixed point

    Data Types: double

<div align="center">
    <img src="Assets/21.jpg" height="250" width="250">
    <img src="Assets/22.jpg" height="250" width="250">
</div>

```python
from math import log10, sqrt
import cv2
import numpy as np
  
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.          
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
  
def main():
     original = cv2.imread("original_image.png")
     compressed = cv2.imread("compressed_image.png", 1)
     value = PSNR(original, compressed)
     print(f"PSNR value is {value})
       
if __name__ == "__main__":
    main()
```

    Output: 
    PSNR value is 43.862955653517126
    
    *Note: Above code and mentioned images are different, demonstrated for just of understanding 

---

__📌 SSIM__

__The Structural Similarity Index (SSIM)__ is a perceptual metric that quantifies the image quality degradation that is caused by processing such as data compression or by losses in data transmission. This metric is basically a full reference that requires 2 images from the same shot, this means 2 graphically identical images to the human eye. The second image generally is compressed or has a different quality, which is the goal of this index.

- SSIM actually measures the perceptual difference between two similar images.
- Generally SSIM values 0.97, 0.98, 0.99 for good quallty recontruction techniques.

<div align="center">
    <img src="Assets/23.jpg" height="250" width="500">
</div>

```python
from skimage.metrics import structural_similarity
import imutils
import cv2
from PIL import Image

original = cv2.imread('pan_card_tampering/image/original.png')
tampered = cv2.imread('pan_card_tampering/image/tampered.png')

# The file format of the source file.
print("Original image format : ",original.format) 
print("Tampered image format : ",tampered.format)
# Image size, in pixels. The size is given as a 2-tuple (width, height).
print("Original image size : ",original.size) 
print("Tampered image size : ",tampered.size)

# Resize Image
original = original.resize((250, 160))
print(original.size)
original.save('pan_card_tampering/image/original.png')#Save image
tampered = tampered.resize((250,160))
print(tampered.size)
tampered.save('pan_card_tampering/image/tampered.png')#Saves image

# Convert the images to grayscale
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
tampered_gray = cv2.cvtColor(tampered, cv2.COLOR_BGR2GRAY)

(score, diff) = structural_similarity(original_gray, tampered_gray, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM Score is : {}".format(score*100))
if score >= 80:
    print ("The given pan card is original")
else:
    print("The given pan card is tampered")

# Calculating threshold and contours 
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

for c in cnts:
    # applying contours on image
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(original, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(tampered, (x, y), (x + w, y + h), (0, 0, 255), 2)

#Display original image with contour
print('Original Format Image')
original_contour = Image.fromarray(original)
original_contour.save("demo/original_contour_image.png")
```

    Output: 
    SSIM Score is : 31.678790332739425 
    The given pan card is tampered

---

## 🧲 Part 4: NLP Evaluation Metrics


  
<div align="left">
<img src="https://user-images.githubusercontent.com/41562231/141720940-53eb9b25-777d-4057-9c2d-8e22d2677c7c.png" height="120" width="900">
</div>
