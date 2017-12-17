---
title: "Practical Machine Learning Assessment"
output:
  html_document:
    highlight: pygments
    keep_md: yes
    theme: united
  pdf_document:
    highlight: zenburn
author: "Partha De"
date: "Sunday, December 17, 2017"

---

## Summary

This report uses machine learning algorithms to predict the manner in which users of exercise devices exercise. 


### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [here:](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset). 

### Data 

The training data for this project are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: 

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


### Set the work environment and knitr options


```r
rm(list=ls(all=TRUE)) #start with empty workspace
startTime <- Sys.time()

library(knitr)
opts_chunk$set(echo = TRUE, cache= TRUE, results = 'hold')
```

### Load libraries and Set Seed

Load all libraries used, and setting seed for reproducibility. *Results Hidden, Warnings FALSE and Messages FALSE*


```r
library(ElemStatLearn)
library(caret)
library(rpart)
library(randomForest)
library(RCurl)
set.seed(2017)
```

### Load and prepare the data and clean up the data




Load and prepare the data


```r
trainingLink <- getURL("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
pml_CSV  <- read.csv(text = trainingLink, header=TRUE, sep=",", na.strings=c("NA",""))

pml_CSV <- pml_CSV[,-1] # Remove the first column that represents a ID Row
```

### Data Sets Partitions Definitions

Create data partitions of training and validating data sets.


```r
inTrain = createDataPartition(pml_CSV$classe, p=0.60, list=FALSE)
training = pml_CSV[inTrain,]
validating = pml_CSV[-inTrain,]

# number of rows and columns of data in the training set

dim(training)

# number of rows and columns of data in the validating set

dim(validating)
```

```
## [1] 11776   159
## [1] 7846  159
```
## Data Exploration and Cleaning

Since we choose a random forest model and we have a data set with too many columns, first we check if we have many problems with columns without data. So, remove columns that have less than 60% of data entered.


```r
# Number of cols with less than 60% of data
sum((colSums(!is.na(training[,-ncol(training)])) < 0.6*nrow(training)))
```

[1] 100

```r
# apply our definition of remove columns that most doesn't have data, before its apply to the model.

Keep <- c((colSums(!is.na(training[,-ncol(training)])) >= 0.6*nrow(training)))
training   <-  training[,Keep]
validating <- validating[,Keep]

# number of rows and columns of data in the final training set

dim(training)
```

[1] 11776    59

```r
# number of rows and columns of data in the final validating set

dim(validating)
```

[1] 7846   59

## Modeling
In random forests, there is no need for cross-validation or a separate test set to get an unbiased estimate of the test set error. It is estimated internally, during the execution. So, we proceed with the training the model (Random Forest) with the training data set.


```r
model <- randomForest(classe~.,data=training)
print(model)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.19%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3346    2    0    0    0 0.0005973716
## B    2 2277    0    0    0 0.0008775779
## C    0    4 2048    2    0 0.0029211295
## D    0    0    8 1921    1 0.0046632124
## E    0    0    0    3 2162 0.0013856813
```

### Model Evaluate
And proceed with the verification of variable importance measures as produced by random Forest:


```r
importance(model)
```

```
##                      MeanDecreaseGini
## user_name                  84.3141523
## raw_timestamp_part_1      928.4162310
## raw_timestamp_part_2       10.2897094
## cvtd_timestamp           1424.2970949
## new_window                  0.2128058
## num_window                567.0564362
## roll_belt                 536.5276164
## pitch_belt                289.9382245
## yaw_belt                  327.3182234
## total_accel_belt          108.4458620
## gyros_belt_x               37.7515708
## gyros_belt_y               48.0117795
## gyros_belt_z              125.3598640
## accel_belt_x               68.4523640
## accel_belt_y               69.1136648
## accel_belt_z              186.6227253
## magnet_belt_x             106.9151200
## magnet_belt_y             193.4659051
## magnet_belt_z             184.9507754
## roll_arm                  123.3551293
## pitch_arm                  54.8725746
## yaw_arm                    73.1475598
## total_accel_arm            28.9628891
## gyros_arm_x                44.9883639
## gyros_arm_y                46.3333452
## gyros_arm_z                18.0820146
## accel_arm_x                86.9412863
## accel_arm_y                54.3746419
## accel_arm_z                38.7249039
## magnet_arm_x               96.6932980
## magnet_arm_y               84.0878499
## magnet_arm_z               58.3163752
## roll_dumbbell             201.3950708
## pitch_dumbbell             82.4647427
## yaw_dumbbell              110.9708329
## total_accel_dumbbell      117.5420803
## gyros_dumbbell_x           44.7133995
## gyros_dumbbell_y          111.7835980
## gyros_dumbbell_z           27.3022178
## accel_dumbbell_x          130.3573255
## accel_dumbbell_y          183.7565946
## accel_dumbbell_z          140.4126094
## magnet_dumbbell_x         237.9440879
## magnet_dumbbell_y         319.2256493
## magnet_dumbbell_z         279.4442908
## roll_forearm              231.1148799
## pitch_forearm             295.2674741
## yaw_forearm                55.0372563
## total_accel_forearm        33.6252832
## gyros_forearm_x            24.5897579
## gyros_forearm_y            38.6290287
## gyros_forearm_z            26.5432786
## accel_forearm_x           133.2561899
## accel_forearm_y            44.6569685
## accel_forearm_z            86.3973162
## magnet_forearm_x           75.5175666
## magnet_forearm_y           73.1377617
## magnet_forearm_z           97.6090769
```

Now we evaluate our model results through confusion Matrix.


```r
confusionMatrix(predict(model,newdata=validating[,-ncol(validating)]),validating$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    0    0    0    0
##          B    1 1518    4    0    0
##          C    0    0 1363    1    0
##          D    0    0    1 1285    1
##          E    0    0    0    0 1441
## 
## Overall Statistics
##                                          
##                Accuracy : 0.999          
##                  95% CI : (0.998, 0.9996)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9987         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   1.0000   0.9963   0.9992   0.9993
## Specificity            1.0000   0.9992   0.9998   0.9997   1.0000
## Pos Pred Value         1.0000   0.9967   0.9993   0.9984   1.0000
## Neg Pred Value         0.9998   1.0000   0.9992   0.9998   0.9998
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1737   0.1638   0.1837
## Detection Prevalence   0.2843   0.1941   0.1738   0.1640   0.1837
## Balanced Accuracy      0.9998   0.9996   0.9981   0.9995   0.9997
```

And confirmed the accuracy at validating data set by calculate it with the formula:


```r
accuracy <-c(as.numeric(predict(model,newdata=validating[,-ncol(validating)])==validating$classe))

accuracy <-sum(accuracy)*100/nrow(validating)
```

Model Accuracy as tested over Validation set = **99.9%**.  

### Model Test

Finally, we proceed with predicting the new values in the testing csv provided, first we apply the same data cleaning operations on it and coerce all columns of testing data set for the same class of previous data set. 

#### Getting Testing Dataset


```r
testingLink <- getURL("http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
pml_CSV  <- read.csv(text = testingLink, header=TRUE, sep=",", na.strings=c("NA",""))

pml_CSV <- pml_CSV[,-1] # Remove the first column that represents a ID Row
pml_CSV <- pml_CSV[ , Keep] # Keep the same columns of testing dataset
pml_CSV <- pml_CSV[,-ncol(pml_CSV)] # Remove the problem ID

# Apply the Same Transformations and Coerce Testing Dataset

# Coerce testing dataset to same class and strucuture of training dataset 
testing <- rbind(training[100, -59] , pml_CSV) 
# Apply the ID Row to row.names and 100 for dummy row from testing dataset 
row.names(testing) <- c(100, 1:20)
```

#### Predicting with testing dataset


```r
predictions <- predict(model,newdata=testing[-1,])
print(predictions)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

#### The following function to create the files to answers the Prediction Assignment Submission:


```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0(pathAnswers,"answers/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(predictions)

#get the time
```


```r
endTime <- Sys.time()
```
The analysis was completed on Sun Dec 17 9:42:27 PM 2017  in 1 seconds.
