
#Clean data

bc_data <- read.table("C:/Users/Manon/Documents/Cours UTC/GB06/ML01/Project/breast-cancer-wisconsin.data", header = FALSE, sep = ",")
head(bc_data)

colnames(bc_data) <- c("sample_code_number", "clump_thickness", "uniformity_of_cell_size", "uniformity_of_cell_shape", "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei", "bland_chromatin", "normal_nucleoli", "mitosis", "classes")

bc_data$classes.num=bc_data$classes # we copie the class column and name it classes.num
bc_data$classes.num[bc_data$classes.num=="2"] <- 0
bc_data$classes.num[bc_data$classes.num=="4"] <- 1
bc_data$classes[bc_data$classes=="2"] <- "benign"
bc_data$classes[bc_data$classes=="4"] <- "malignant"
head(bc_data)

bc_data[bc_data=="?"]<-NA
nrow(bc_data)-length(complete.cases(bc_data)[complete.cases(bc_data)==TRUE])
length(bc_data$bare_nuclei[is.na(bc_data$bare_nuclei)])

bc_data_sansna<-na.omit(bc_data)

# We want to predict the bare nuclei

data<-bc_data_sansna[,2:10]

#on va essayer de prédire toute la colonne bare nuclei comme ça 

library('FNN')

for (i in 1:ncol(data)){
  if (class(data[5,i])=='factor'){
    data[,i] = as.numeric(data[,i])
    print(c(i, "DONE"))
    
  }
  print(c(i,class(data[,i])))
}

n<-nrow(data)
ntrain<-round(2*n/3)
ntest<-n-ntrain
train<-sample(n,ntrain)
bc_data.train<-data[train,]
bc_data.train2<-scale(bc_data.train)
y.train<-bc_data.train$bare_nuclei
bc_data.test<-data[-train,]
bc_data.test2<-scale(bc_data.test)
y.test<-bc_data.test$bare_nuclei


"x.train<-scale(data[data$train==T,1:4]) #on garde que les colonnes 1,2,3 et4 où train = true
summary(x.train)
y.train<-data[data$train==T,5]# que la colonne lpsa où train = true 
y.train
x.tst<-scale(data[data$train==F,1:4])#les 4 premieres colonnes où train =F
y.tst<-data[data$train==F,5]# que la colonne lpsa où train = f"

class(bc_data[,2])

reg<-knn.reg(train=bc_data.train2, test = bc_data.test2, y=y.train, k = 5)# on applique le model knn en se basant sur le training valeurs on fait les test sur la partie réservée. y est la réponse des valeur train

mean((y.test-reg$pred)^2) # = Mean squared error 
plot(y.test,reg$pred,xlab='y',ylab='prediction') 
#si les point sont sur la ligne x=y c'est que le aleur prédite est la même que la vraie valuer 
abline(0,1)




reg<- lm(bare_nuclei ~. , data=bc_data.train)
reg
pred<-predict(reg,newdata=bc_data.test)
plot(y.test,pred)
abline(0,1)

mse<-mean((y.test-pred)^2)

