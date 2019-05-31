---
##  title: "ML01-project"
## output: html_notebook
---
  "
  Breast Cancer data analysis
  
  Introduction
  The phenotypes for characterisation are:
  
Sample ID (code number)
Clump thickness
Uniformity of cell size
Uniformity of cell shape
Marginal adhesion
Single epithelial cell size 
Number of bare nuclei
Bland chromatin
Number of normal nuclei
Mitosis
Classes, i.e. diagnosis

 clean the data
```{r}"
bc_data <- read.table("C:/Users/Manon/Documents/Cours UTC/GB06/ML01/Project/breast-cancer-wisconsin.data", header = FALSE, sep = ",")
head(bc_data)

"
#As we can see, the dataset has no header, so we add the collum names in order to have an easier manipulation of the data.
#```{r}"

colnames(bc_data) <- c("sample_code_number", "clump_thickness", "uniformity_of_cell_size", "uniformity_of_cell_shape", "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitosis", "classes")

#We also want to change the name for the response variable

bc_data$classes.num=bc_data$classes # we copie the class column and name it classes.num
bc_data$classes.num[bc_data$classes.num=="2"] <- 0
bc_data$classes.num[bc_data$classes.num=="4"] <- 1
bc_data$classes[bc_data$classes=="2"] <- "benign"
bc_data$classes[bc_data$classes=="4"] <- "malignant"
head(bc_data)
#```
#```{r}
bc_data[bc_data=="?"]<-NA
#length(bc_data[bc_data$bare_nuclei==NA])
nrow(bc_data)-length(complete.cases(bc_data)[complete.cases(bc_data)==TRUE])
#complete.cases = return vector indicating chich cases are complete 
#return the number of row 
length(bc_data$bare_nuclei[is.na(bc_data$bare_nuclei)])
#is.na= indicate which element are missing

"```
We have 16 missing data that are all in the bc_data$bare_nuclei collunm.
As the number of observation with missing data is low compared to the total number of observation, we could just ignore these 16 observations for the rest of the study and loose small amount of data. However, we could also replace the missing values by the mean of the collum. It does not change the global mean but reduces the variance.
We can also try to apply some algorith that will guess the value with the MICE library. We could also use the library Amelia but we have to make the asomption that all variables follow a multivariate law.
```{r}"

bc_data<-na.omit(bc_data) #we remove the 16 rows with NA 
bc_data[,2:10] <- apply(bc_data[, 2:10], 2, function(x) as.numeric(as.character(x)))#?
#bc_data$classes <- as.factor(bc_data$classes)
#bc_data$classes <- as.numeric(bc_data$classes)
#nrow(bc_data)
"
Now we have a clean dataset.

## Visualisation
```{r}"
summary(bc_data)
#```
#```{r}
plot(bc_data[,2:10],col=(bc_data$classes.num+1))
"```
We dont add the response variable into the scatter plot because box plots are better to display binnary variables

```{r}"
attach(mtcars)
par(mfrow=c(3,3))
for (c in c(2:10)){
  boxplot(bc_data[,c]~classes,data=bc_data,xlab="classes",ylab=colnames(bc_data)[c])
}
```
#Make two boxplot per graph

## Correlation
```{r}
library(corrplot)
```
```{r}
# calculate correlation matrix
corMatMy <- cor(bc_data[, c(-1,-11)])
#corMatMy <- cor(bc_data[, -1])
## Warning : we need to delete old plot before 
corrplot(corMatMy, order = "hclust")
corrplot(corMatMy, order = "hclust", method="number")


"## PCA
### what is PCA?

## Classification
### why do we do classification?

Choose a classifier C(X) that assigns a class label from {benign; malignant} to a future unlabeled observation X.
We want to assess the uncertainty in each classification
we want to understand the roles of the different predictors.

### Create train subset
We divide the data into test and train dataset. We don't need to use the function scale() because the data uses already the same range ( from 1 to 10)
"



#Aplication PCA

breast<-scale(bc_data[,2:10])
pca<-princomp(breast,cor=TRUE)
Z_breast<-pca$scores
lambda2<-pca$sdev^2

plot(cumsum(lambda2)/sum(lambda2),type="b",
     xlab="q",ylab="proportion of explained variance")

biplot(pca)

pca$loadings

# regarder d'autres explications 

### Bayes Classifier

#This probability is called the Bayes error rate. It is the lowest error probability that can be achieved by a classifier. It characterizes the difficulty of the classification task

### K nearest neighbor
#### facts
"We can say that half of the information provided by the training set is contained in the nearest neighbor (asymptotically).
Howerver, the KNN classifier breaks down for big dimention of datasets. We have 10 predictors and that is already a lot for KNN."
#### implementation
#```{r}

library(FNN)
#```

#```{r}

n<-nrow(bc_data)
ntrain<-round(2*n/3)
ntest<-n-ntrain
train<-sample(n,ntrain)
bc_data.train<-bc_data[train,]
bc_data.test<-bc_data[-train,]

classespred<-knn(bc_data.train[,2:10],bc_data.test[,2:10],factor(bc_data.train$classes),k=5)
table(bc_data.test$classes,classespred)#adjacent error matrix
#```
#```{r}
mean(bc_data.test$classes!=classespred)
#```
#### find the optimal k
#```{r}
Kmax<-15
ERR100<-c(0,Kmax)
for(k in 1:Kmax){
classespred<-knn(bc_data.train[,2:10],bc_data.test[,2:10],factor(bc_data.train$classes),k=k)
ERR100[k]<-mean(bc_data.test$classes!=classespred)
}
plot(1:Kmax,ERR100,type = "b")
#```
#```{r}
k_best <- which.min(ERR100)
k_best
#```
#### box plot of optimal k
#```{r}
Kmax<-15
max_sim<-20
K_best<-c(0,max_sim)
for(s in 1:max_sim){
ERR100<-c(0,Kmax)
train<-sample(n,ntrain)
bc_data.train<-bc_data[train,]
bc_data.test<-bc_data[-train,]
for(k in 1:Kmax){
classespred<-knn(bc_data.train[,2:10],bc_data.test[,2:10],factor(bc_data.train$classes),k=k)
ERR100[k]<-mean(bc_data.test$classes!=classespred)
}
k_best[s] <- which.min(ERR100)
}
boxplot(k_best,ylab="optimal k")
#```

###  Linear Discriminant Analysis (LDA)
####Theory
"The parameters to estimates are ??k,^ ??k,and ^??

LDA uses the full likelihood based on the joint distribution of X and Y (generative model).

LDA assumes that the data are *Gaussian*. More specifically, it assumes that all classes share the same covariance matrix.

LDA finds linear decision boundaries in a K???1
dimensional subspace. As such, it is not suited if there are higher-order interactions between the independent variables.
Although LDA also has a number of parameters proportion top2, it isusually much more stable than QDA. This method is recommended when n is small.
"

####Implementation
#```{r}
#bc_data.train.scaled<-scale(bc_data[train,c(-1,-11)])
#bc_data.test.scaled<-scale(bc_data[-train,c(-1,-11)])
#```

#```{r}
library(MASS)
lda.bc_data <- lda(classes~. ,data=bc_data.train[,c(-1,-12)])
#lda.bc_data
pred.bc_data.lda<-predict(lda.bc_data,newdata=bc_data.test[,c(-1,-12)])
perf <-table(bc_data.test$classes,pred.bc_data.lda$class)
print(perf)#confusion matrix
1-sum(diag(perf))/nrow(bc_data.test)#mean error rate
#```
####Receiver Operating Characteristic (ROC)
"In our case we have two classes therefore LDA assigns x to class 2 if xT^?????1(^??2???^??1 ) >s,where the threshold *s* depends on the estimated prior probabibitie. If the prior probabilities cannot be estimated, or if the model assumption are not verified, a different threshold may give betterresult.
The Receiver Operating Characteristic (ROC) curve describes theperformance of the classifier for any value of s. By changing the value of s we can change the confusion matrix values. This allow us to prioritize the true positive rate (sensitivity) and false positive rate (1-specificity).
```{r}"
library(pROC)
roc_lda<-roc(bc_data.test$classes,as.vector(pred.bc_data.lda$x))
plot(roc_lda)
#```
### Quadratic Discriminant Analysis (QDA)
#LDA don't assumes that all classes share the same covariance matrix. So The parameters to estimates are ??k,^ ??k,and ^??k. Mean between classes, mean inside classe and empirical variance matrix in class k.

####Implementation
#```{r}
qda.bc_data <- qda(classes~. ,data=bc_data.train[,c(-1,-12)])
pred.bc_data.qda<-predict(qda.bc_data,newdata=bc_data.test[,c(-1,-12)])
perf <-table(bc_data.test$classes,pred.bc_data.qda$class)
print(perf)#confusion matrix
1-sum(diag(perf))/nrow(bc_data.test)#mean error rate
#```

#QDA pas assez stage LDA est meux quand p est petit 

### Naive Bayes
####Theory
"For Naive Bayes classifers, we set the covariance matrix to diagonal matrix. This assumption means that the predictors are conditionally independent given the class variable  Y. A further simplification is achieved by assuming that the covariancematrices are diagonal and equal:??1=···=??c=??=diag(??21,...,??2p).
Naive Bayes classifiers have a number of parameters proportional to p.They usually outperform other methods when p is very large.
"
#### Implementation
#```{r}
library(naivebayes)
nb.bc_data <- naive_bayes(as.factor(classes)~. ,data=bc_data.train[,c(-1,-12)])
pred.bc_data.nb<-predict(nb.bc_data,newdata=bc_data.test[,c(-1,-12)],type="class")
perf <-table(bc_data.test$classes,pred.bc_data.nb)
print(perf)#confusion matrix
1-sum(diag(perf))/nrow(bc_data.test)#mean error rate
#```


### Logistic regression
"Logistic regression uses the conditional likelihood based on the conditional probabilities Pk(x) (discriminative model).
logReg models fit by maximizing the conditional likelihood, which is the likelihood function, assuming thexiare fixed. unction`(??) =logL(??)is maximized using an iterative optimizationalgorithm: the Newton-Raphson algorithm.
"
#### Binomial logistic regression
#As we have only two classes to predict, we can use this classifier
#```{r}
glm.bc_data <- glm(as.factor(classes)~. ,data=bc_data.train[,c(-1,-12)],family=binomial)
pred.bc_data.glm<-predict(glm.bc_data,newdata=bc_data.test[,c(-1,-12)],type="response")
perf <-table(bc_data.test$classes,pred.bc_data.glm>0.5)
print(perf)#confusion matrix
1-sum(diag(perf))/nrow(bc_data.test)#mean error rate
#```
### Tree classifier
#### Impurity measures Qt

"Unlike the tree for the regression, we will not use Qt= MSE within the region. We have three other options:
  Misclassification error, Gini Index, Entropy.
In our casse we have two classes: if p is the proportion of malign in the region Rt then: 
  Misclassification error = 1???max(p,1???p),
Gini Index =  2p(1???p),
Entropy = ???plogp???(1???p)log(1???p).
All three are similar, but entropy and the Gini index are differentiable,and hence more amenable to numerical optimization.

Consider a nodetwith sizentwith impurit yQt. For some variable j and split point s, we split t in two nodes, tL and tR, with sizes ntL and ntR, and with impurities QtL and QtR.
The average decrease of impurity is ???(j,s) =Qt???(ntLntQtL+ntRntQtR) If Qt is the entropy, then ???(j,s)is interpreted as an information gain.We select at each step the splitting variablejand the split pointsthatmaximizes???(j,s)or, equivalently, that minimizes the average impurity.

When splitting a predictor having q possible unordered values, thereare 2q???1???1 possible partitions of the q values into two groups.
All the dichotomies can be explored for small q, but the computations become prohibitive for large q. In the 2-class case, this computation simplifies. We order the predictorlevels according to the proportion falling in outcome class 1. Then wesplit this predictor as if it were an ordered predictor. One can show this gives the optimal split, in terms of entropy or Gini index, among all possible 2q???1???1 splits.

Trees can easily handle qualitative predictors without the need tocreate dummy variables. But the algorith of partitionning tends to favor predictor with many levels: they should be avoided. In our case this is not a problem because every predicto has the same number of level.
"

#### Implementation

library(rpart)
tree.bc_data<-rpart(classes~.,data=bc_data.train[2:11],method="class",control=rpart.control(xval = 10,minbucket = 5,cp=0))
#x val is number of cross-validations
#minbucket is the minimum number of observations in any terminal <leaf> node.
# cp is complexity parameter. Any split that does not decrease the overall lack of fit by a factor of cp is not attempted.
plot(tree.bc_data,margin = 0.05)
text(tree.bc_data,pretty=0,cex=0.8)

pred.bc_data.tree=predict(tree.bc_data,newdata=bc_data.test,type='class')
table(bc_data.test[,"classes"],pred.bc_data.tree)
err<-mean(bc_data.test[,"classes"]!=pred.bc_data.tree)
print(err)


plotcp(tree.bc_data)
printcp(tree.bc_data)

#We see that for a complexity of 0.015, we have the smallest cross validation relative error
#### Pruning
#```{r}
pruned_tree<-prune(tree = tree.bc_data,cp=0.036810)
plot(pruned_tree,margin = 0.1)
text(pruned_tree,pretty=0)
pred.pruned_tree=predict(pruned_tree,newdata=bc_data.test[2:11],type='class')
table(bc_data.test[,"classes"],pred.pruned_tree)
err1<-mean(bc_data.test[,"classes"]!=pred.pruned_tree)
print(err1)
#```
#As we can see, by prunning, we have a different confusion matrix, a lower complexity but the same error rate.
#### Bagging
#As we can see, trees generally do not have the same level of predictiveaccuracy as some of the other modern regression and classificationapproaches. To improve the quality of the prediction, we can use bagging method.

##Model selection 

library(glmnet)
#library(bestglm)
library(leaps)
library(nnet)

#Préparation des données 

bc_data.2<-bc_data[,2:11]
n<-nrow(bc_data.2)
ntrain<-round(2*n/3)
ntest<-n-ntrain
train<-sample(n,ntrain)
bc_data.train<-bc_data.2[train,]#on a pas la premiere colonne
bc_data.test<-bc_data.2[-train,]#on a pas la premiere colonne 

xtst2<-as.matrix(bc_data.test[,1:9])#on enlève la colonne des résultats 
ntst2<-nrow(xtst2)
X2<-cbind(rep(1,ntst2),xtst2)
ytst2<-bc_data.test$classes #on a les réponses (=classes) pour les observations non entrainées 


#reg.forward<-regsubsets(as.factor(classes)~.,data=bc_data.2,
                        #method='forward',nvmax=5)
reg.forward<-regsubsets(as.factor(classes)~.,data=bc_data.2,
                        method='forward')
plot(reg.forward,scale="bic")
res<-summary(reg.forward)
names(res)

res
res$bic

plot(reg.forward,scale="adjr2")


# BIC

best<-which.min(res$bic)

ypred2<-X2[,res$which[best,]]%*%coef(reg.forward,best)
mse_forward_bic<-mean((ypred2-ytst2)^2)

# Adjusted R2
plot(reg.forward,scale="adjr2")
best<-which.max(res$adjr2)
ypred<-X[,res$which[best,]]%*%coef(reg.forward,best)
mse_forward_adjr2<-mean((ypred-ytst)^2)

# Backward selection 
reg.backward<-regsubsets(lpsa~.,data=prostate[train==TRUE,],method='backward',nvmax=30)
plot(reg.backward,scale="bic")
res<-summary(reg.backward)

bc_data_pronostic <- read.table("C:/Users/Manon/Documents/Cours UTC/GB06/ML01/Project/wpbc.data", header = FALSE, sep = ",")

