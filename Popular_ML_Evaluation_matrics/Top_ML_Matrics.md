<div align="right">
<img src="https://user-images.githubusercontent.com/41562231/141720820-090897f9-f564-45e2-9265-15c1269db795.png" height="120" width="900">
</div>


# __Popular Machine Learning Metrics.__

__Introduction-__
Choosing the right metric is crucial while evaluating machine learning (ML) models. Various metrics are proposed to evaluate ML models in different applications, and I thought it may be helpful to provide a summary of popular metrics in a here, for better understanding of each metric and the applications they can be used for. In some applications looking at a single metric may not give you the whole picture of the problem you are solving, and you may want to use a subset of the metrics discussed in this post to have a concrete evaluation of your models.

- __Classification Metrics__ (accuracy, precision, recall, F1-score, ROC, AUC, log-loss)
- __Regression Metrics__ (MSE, MAE, RMSE, Adj R2. R2)
- __Ranking Metrics__ (MRR, DCG, NDCG)
- __Statistical Metrics__ (Correlation)
- __Computer Vision Metrics__ (PSNR, SSIM, IoU)
- __NLP Metrics__ (Perplexity, BLEU score)
- __Deep Learning Related Metrics__ (Inception score, Frechet Inception distance)

## 🧲 Part 1: Classification & Regression Evaluation Metrics
    A metric is different from loss function?

    Loss functions are functions that show a measure of the model performance and are used to train a machine learning model (using some kind of optimization), and are usually differentiable in model’s parameters. On the other hand, metrics are used to monitor and measure the performance of a model (during training, and test), and do not need to be differentiable. However if for some tasks the performance metric is differentiable, it can be used both as a loss function (perhaps with some regularizations added to it), and a metric, such as MSE.

<div align="center">
    <h2><b>🦮 Classiication Related Metrics</b></h2>
    Models such as support vector machine (SVM), logistic regression, decision trees, random forest, XGboost, convolutional neural network¹, recurrent neural network are some of the most popular classification models
</div><br>

__📌 Confusion Matrix:__ _(not a metric, but important to know!)_

One of the key concept in classification performance is confusion matrix (AKA error matrix), which is a tabular visualization of the model predictions versus the ground-truth labels. Each row of confusion matrix represents the instances in a predicted class and each column represents the instances in an actual class.

img 1

- __Out of 100 cat images__ the model has predicted 90 of them correctly and has mis-classified 10 of them. If we refer to the “cat” class as positive and the non-cat class as negative class, then 90 samples predicted as cat are considered as as __true-positive__, and the 10 samples predicted as non-cat are __false negative__.
- __Out of 1000 non-cat images__, the model has classified 940 of them correctly, and mis-classified 60 of them. The 940 correctly classified samples are referred as __true-negative__, and those 60 are referred as __false-positive__.

_Another Example_

fig 10

- _True Positive:_ We predicted positive and it’s true. In the image, we predicted that a woman is pregnant and she actually is.
- _True Negative:_ We predicted negative and it’s true. In the image, we predicted that a man is not pregnant and he actually is not.
- _False Positive (Type 1 Error)_- We predicted positive and it’s false. In the image, we predicted that a man is pregnant but he actually is not.
- _False Negative (Type 2 Error)_- We predicted negative and it’s false. In the image, we predicted that a woman is not pregnant but she actually is.

__📌 Classification Accuracy:__

Classification accuracy is perhaps the simplest metrics one can imagine, and is defined as __the number of correct predictions divided by the total number of predictions__, multiplied by 100. So in the above example, out of 1100 samples 1030 are predicted correctly, resulting in a classification accuracy of:

fig 9

    Classification accuracy= (90+940)/(1000+100)= 1030/1100= 93.6%

__📌 Precision:__

There are many cases in which classification accuracy is not a good indicator of your model performance. One of these scenarios is when your class distribution is imbalanced (one class is more frequent than others). In this case, even if you predict all samples as the most frequent class you would get a high accuracy rate, which does not make sense at all (because your model is not learning anything, and is just predicting everything as the top class). For example in our cat vs non-cat classification above, if the model predicts all samples as non-cat, it would result in a 1000/1100= 90.9%. 

__Precision = True_Positive/ (True_Positive+ False_Positive)__

fig 11

    Precision_cat = samples correctly predicted cat/samples predicted as cat 
                  = 90/(90+60) 
                  = 60%
    Precision_NonCat = 940/950
                     = 98.9%

__📌 Recall:__

Recall is another important metric, which is defined as the fraction of samples from a class which are correctly predicted by the model.

__Recall = True_Positive/ (True_Positive+ False_Negative)__
    
fig 12

    Recall_cat = 90/100
               = 90%
    Recall_NonCat = 940/1000
                  = 94%

__📌 F1 Score:__

Depending on application, you may want to give higher priority to recall or precision. But there are many applications in which both recall and precision are important. Therefore, it is natural to think of a way to combine these two into a single metric. One popular metric which __combines precision and recall is called F1-score__, which is the harmonic mean of precision and recall

__F1-score= 2*Precision*Recall/(Precision+Recall)__

fig 13

    F1_cat = 2*0.6*0.9/(0.6+0.9)
           = 72%

_Note:_ if you want to make the precision too high, you would end up seeing a drop in the recall rate, and vice versa.

__📌 Sensitivity and Specificity:__
Sensitivity and specificity are two other popular metrics mostly used in medical and biology related fields which are very sensitive to the data.

__Sensitivity = Recall= TP/(TP+FN)__

__Specificity = True Negative Rate= TN/(TN+FP)__

__📌 ROC Curve:__
The receiver operating characteristic curve is plot which shows the performance of a binary classifier as function of its cut-off threshold. It essentially shows the __true positive rate (TPR) against the false positive rate (FPR)__ for various threshold values.

- Many of the classification models are probabilistic, i.e. they predict the probability of a sample being a cat. They then compare that output probability with some cut-off threshold and if it is larger than the threshold they predict its label as cat, otherwise as non-cat. 
- As an example your model may predict the below probabilities for __4 sample images: [0.45, 0.6, 0.7, 0.3]__. Then depending on the __threshold values__ below, you will get different labels:

        cut-off= 0.5: predicted-labels= [0,1,1,0] (default threshold)
        cut-off= 0.2: predicted-labels= [1,1,1,1]
        cut-off= 0.8: predicted-labels= [0,0,0,0]

        ROC curve essentially finds out the TPR and FPR for various threshold values and plots TPR against the FPR

img 2

- From figure, the lower the cut-off threshold on positive class, the more samples predicted as positive class, i.e. higher true positive rate (recall) and also higher false positive rate (corresponding to the right side of this curve). Therefore, there is a trade-off between how high the recall could be versus how much we want to bound the error (FPR).

__📌 AUC:__

The __Area Under the Curve (AUC)__, is an aggregated measure of performance of a binary classifier on all possible threshold values (and therefore it is threshold invariant). AUC calculates the area under the ROC curve, and therefore it is between 0 and 1. One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example

fig 3

- On high-level, the higher the AUC of a model the better it is. But sometimes threshold independent measure is not what you want, e.g. you may care about your model recall and require that to be higher than 99% (while it has a reasonable precision or FPR). In that case, you may want to tune your model threshold such that it meets your minimum requirement on those metrics (and you may not care even if you model AUC is not too high).
- Therefore in order to decide how to evaluate your classification model performance, perhaps you want to have a good understanding of the business/problem requirement and the impact of low recall vs. low precision, and decide what metric to optimize for.
- From a practical standpoint, a classification model which outputs probabilities is preferred over a single label output, as it provides the flexibility of tuning the threshold such that it meets your minimum recall/precision requirements. Not all models provide this nice probabilistic outputs though, e.g. SVM does not provide a simple probability as an output (although it provides margin which can be used to tune the decision, but it is not as straightforward and interpretable as having output probabilities).

__📌 Log Loss:__ 

Log loss (Logistic loss) or Cross-Entropy Loss is one of the major metrics to assess the performance of a classification problem.

<div align="center">
    <h2><b>📈 Regression Related Metrics</b></h2>
    Regression models are another family of machine learning and statistical models, which are used to predict a continuous target values.
</div><br>

__📌 MSE:__

__Mean squared Error__ is perhaps the most popular metric used for regression problems. It essentially finds the average squared error between the predicted and actual values. It represents the squared distance between actual and predicted values. we perform squared to avoid the cancellation of negative terms and it is the benefit of MSE.

fig 4

- Advantages of MSE:
    - The graph of MSE is differentiable, so you can easily use it as a loss function.

- Disadvantages of MSE:
    - The value you get after calculating MSE is a squared unit of output. for example, the output variable is in meter(m) then after calculating MSE the output we get is in meter squared.
    - If you have outliers in the dataset then it penalizes the outliers most and the calculated MSE is bigger. So, in short, It is not Robust to outliers which were an advantage in MAE.

__📌 MAE:__

__Mean Absolute Error__ _(mean absolute deviation)_ is another metric which finds the average absolute distance between the predicted and target values. __MAE is known to be more robust to the outliers than MSE.__ The main reason being that in MSE by squaring the errors, the outliers (which usually have higher errors than other samples) get more attention and dominance in the final error and impacting the model parameters.

find the difference between the actual value and predicted value that is an absolute error but we have to find the mean absolute of the complete dataset. Hence sum all the errors and divide them by a total number of observations And this is MAE. And we aim to get a minimum MAE because this is a loss.

fig 5

- Advantages of MAE:
    - The MAE you get is in the same unit as the output variable.
It is most Robust to outliers.
- Disadvantages of MAE:
    - The graph of MAE is not differentiable so we have to apply various optimizers like Gradient descent which can be differentiable.

__📌 RMSE:__

__Root Mean Squared Error__ is clear by the name itself, that it is a simple square root of mean squared error.

fig 6

- Advantages of RMSE:
    - The output value you get is in the same unit as the required output variable which makes interpretation of loss easy.
- Disadvantages of RMSE:
    - It is not that robust to outliers as compared to MAE.

__📌 RMSLE:__

__Root Mean Squared Log Error__ Taking the log of the RMSE metric slows down the scale of error. _The metric is very helpful when you are developing a model without calling the inputs._ In that case, the output will vary on a large scale.

__📌 R Squared (R²):__

R2 score is a metric that tells the performance of your model, not the loss in an absolute sense that how many wells did your model perform.

fig 7

- If the R2 score is zero then the above regression line by mean line is equal means 1 so 1-1 is zero. So, in this case, both lines are overlapping means model performance is worst, It is not capable to take advantage of the output column.
- when the R2 score is 1, it means when the division term is zero and it will happen when the regression line does not make any mistake, it is perfect. In the real world, it is not possible.

__📌 Adjusted R Squared:__

The disadvantage of the R2 score is while adding new features in data the R2 score starts increasing or remains constant but it never decreases because It assumes that while adding more data variance of data increases. But the problem is when we add an irrelevant feature in the dataset then at that time R2 sometimes starts increasing which is incorrect. Hence, To control this situation Adjusted R Squared came into existence.

fig 8

## 🧲 Part 2: Ranking, & Statistical Evaluation Metrics

<div align="center">
    <h2><b>🦮 Ranking Related Metrics</b></h2>
    Ranking is a fundamental problem in machine learning, which tries to rank a list of items based on their relevance in a particular task (e.g. ranking pages on Google based on their relevance to a given query). 
    
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

__📌 MRR:__

Mean reciprocal rank (MRR) is one of the simplest metrics for evaluating ranking models. MRR is essentially the average of the reciprocal ranks of “the first relevant item” for a set of queries Q

fig 14

let’s consider the below example, in which the model is trying to predict the plural form of English words by masking 3 guess. In each case, the correct answer is also given.

fig 15

__The MRR of this system can be found as:__
    
    MRR = 1/3*(1/2+1/3+1/1)
        = 11/18

    Note: One of the limitations of MRR is that, it only takes the rank of one of the items (the most relevant one) into account, and ignores other items (for example mediums as the plural form of medium is ignored). This may not be a good metric for cases that we want to browse a list of related items.

__📌 Precision at k:__

The number of relevant documents among the top k documents

fig 16

- As an example, if you search for “hand sanitizer” on Google, and in the first page, 8 out of 10 links are relevant to hand sanitizer, then the P@10 for this query equals to 0.8.
Now to find the precision at k for a set of queries Q, you can find the average value of P@k for all queries in Q.
- P@k has several limitations. Most importantly, it fails to take into account the positions of the relevant documents among the top k. Also it is easy to evaluate the model manually in this case, since only the top k results need to be examined to determine if they are relevant or not.

__📌 DCG and NDCG:__

Normalized Discounted Cumulative Gain (NDCG) is perhaps the most popular metric for evaluating learning to rank systems. In contrast to the previous metrics, NDCG takes the order and relative importance of the documents into account, and values putting highly relevant documents high up the recommended lists.

Before giving the official definition NDCG, let’s first introduce two relevant metrics, Cumulative Gain (CG) and Discounted Cumulative Gain (DCG).

fig 17

__Normalized Discounted Cumulative Gain (NDCG)__ tries to further enhance DCG to better suit real world applications. Since the retrieved set of items may vary in size among different queries or systems, NDCG tries to compare the performance using the normalized version of DCG (by dividing it by DCG of the ideal system). In other words, it sorts documents of a result list by relevance, finds the highest DCG (achieved by an ideal system) at position p, and used to normalize DCG

_Note: One of its main limitations is that it does not penalize for bad documents in the result. It may not be suitable to measure performance of queries that may often have several equally good results._

<div align="center">
    <h2><b>🦮 Statistical Metrics</b></h2>
    Some of the popular metrics here include: Pearson correlation coefficient, coefficient of determination (R²), Spearman’s rank correlation coefficient, p-value, and more². Here we briefly introduce correlation coefficient, and R-squared
</div><br>

__📌Pearson Correlation Coefficient:__

Pearson correlation coefficient is perhaps one of the most popular metrics in the whole statistics and machine learning area. Its application is so broad that is used in almost every aspects of statistical modeling, from feature selection and dimensionality reduction, to regularization and model evaluation.

Correlation coefficient of two random variables (or any two vector/matrix) shows their statistical dependence.

- The correlation coefficient of two variables is always a __value in [-1,1].__ _Two variables are known to be independent if and only if their __correlation is 0.___

## 🧲 Part 3: Computer Vision Evaluation Metrics

<div align="center">
    <h2><b>🦮 Computer Vision Metrics</b></h2>
    More recently, with the popularization of the convolutional neural networks (CNN) and GPU-accelerated deep-learning frameworks, object- detection algorithms started being developed from a new perspective. CNNs such as R-CNN, Fast R-CNN, Faster R-CNN, R-FCN, SSD and Yolo have highly increased the performance standards on the field.
    
    Object detection metrics serve as a measure to assess how well the model performs on an object detection task. It also enables us to compare multiple detection systems objectively or compare them to a benchmark. In most competitions, the average precision (AP) and its derivations are the metrics adopted to assess the detections and thus rank the teams
</div><br>

__📌 IoU:__

Guiding principle in all state-of-the-art metrics is the so-called __Intersection-over-Union (IoU) overlap measure__. It is quite literally defined as the intersection over union of the detection bounding box and the ground truth bounding box.

Dividing the area of overlap between predicted bounding box and ground truth by the area of their union yields the Intersection over Union.

__An Intersection over Union score > 0.5 is normally considered a good prediction.__

fig 18

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

fig 19

__i) AP -__ The general definition for the Average Precision(AP) is finding the area under the precision-recall curve.

__ii) mAP -__ The mAP for object detection is the average of the AP calculated for all the classes. mAP@0.5 means that it is the mAP calculated at IOU threshold 0.5.

fig 20

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

__📌 PSNR:__

__Peak signal-to-noise ratio (PSNR)__ is the ratio between the maximum possible power of an image and the power of corrupting noise that affects the quality of its representation. If we have 8-bit pixels, then the values of the pixel channels must be from 0 to 255. By the way, the red, green, blue or RGB color model fits best for the PSNR. PSNR shows a ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation.

    Input image, specified as scalar, vector, or matrix.
    
    Data Types: single | double | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 | Boolean | fixed point

    Data Types: double

fig 21, 22
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

__📌 SSIM__

__The Structural Similarity Index (SSIM)__ is a perceptual metric that quantifies the image quality degradation that is caused by processing such as data compression or by losses in data transmission. This metric is basically a full reference that requires 2 images from the same shot, this means 2 graphically identical images to the human eye. The second image generally is compressed or has a different quality, which is the goal of this index.

- SSIM actually measures the perceptual difference between two similar images.
- Generally SSIM values 0.97, 0.98, 0.99 for good quallty recontruction techniques.

fig 23

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
````
    Output: 
    SSIM Score is : 31.678790332739425
    The given pan card is tampered


PSNR, SSIM

  
<div align="left">
<img src="https://user-images.githubusercontent.com/41562231/141720940-53eb9b25-777d-4057-9c2d-8e22d2677c7c.png" height="120" width="900">
</div>
