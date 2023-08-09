##Clustering Analysis prepare data V4
##KS test,t-test,et al. Select Rule
## DA Method
getwd()   					
rm(list = ls())
#setwd('/Users/ksj/Desktop/B00_4FGL/_20190715_RF_Gini')
#setwd('/home/ksj/work/AAS19081_data_v4')
#setwd('/Users/ksj/Desktop/A10_LBL_FSRQ/Data_20200124')
#setwd("/Users/ksj/Desktop/A10_LBL_FSRQ/Data_20200206_4LAC")
#setwd("/Users/ksj/Desktop/A10_LBL_FSRQ_Tex/Data_20200216_4LAC_HR")
#setwd("/Users/ksj/Desktop/A10_B01_4LACv2")
#setwd("/Users/ksj/Desktop/A10_LBL_FSRQ_Tex/Data_20200226_4LAC_HR")
#setwd("/Users/ksj/Desktop/A10_LBL_FSRQ_Tex/Data_20200307_4LAC_HR")
#setwd("/Users/ksj/Desktop/A10_LBL_FSRQ_Tex/Data_20200318_HSP")
#setwd("/Users/ksj/Desktop/4FGL_4LAC/Match_code")
#setwd("/home/ksj/WORK/Data_20201018/A10_DR2")

#setwd("/Users/ksj/Desktop/Data_20220101")
#setwd("/home/kem/HR03_result_72")
#setwd("/home/ksj/WORK/Data_20201018/HR03_result_72")

#setwd("/Users/ksj/Desktop/Data_20220630")

#setwd("/home/kem/KLSPBLLac20220703")


#setwd("/home/kem/G08_high_0719_kem")
#setwd("/Users/ksj/Desktop/Data_20220630/G08_high_0719_kem")
setwd("/Users/ksj/Desktop/Result_good")





getwd()
#dir()
#library(mclust)
par(mfrow = c(1,1))
#options()

# install.packages("stringr",repos="https://mirrors.bfsu.edu.cn/CRAN/")
# install.packages("randomForest",repos="https://mirrors.bfsu.edu.cn/CRAN/")

# install.packages("e1071",repos="https://mirrors.bfsu.edu.cn/CRAN/")

# install.packages("snowfall",repos="https://mirrors.bfsu.edu.cn/CRAN/")

# export OPENBLAS_NUM_THREADS=2
# 
# export GOTO_NUM_THREADS=2
# 
# export OMP_NUM_THREADS=2


library(randomForest)
library(e1071)
library(nnet)



dir()
#####读取数据，准备数据，training set; test set, predict
#FSRQ_BLLAC
#databf32 <- read.csv("CHR03_py_HS_HR_KS_E02_D01_Z_Train_data.csv", header = T)
databf32 <- read.csv("./G01_bu_suan/G07_0718_high_Train_data_X_band.csv",header = T)
dim(databf32)
names(databf32)
databfA <- databf32[,c(3:51)]
dim(databfA)
names(databfA)
table(databfA$CLASS1)
table(databfA$CLASS)
databf <- databfA
dim(databf)

Class <- databf[,c(43)]


#databf <- databfA[,-c(1,46)]
#names(databf)
#databf <- subset(databf_0, curve_significance != 0)
#dim(databf)
#names(databf)
#X <- log10(databf[,c(64:70)])
#Class <- databf[,c(46)]
#names(X);
#names(Class)


#BCU402
#BCU40032 <- read.csv("CHR03_py_HS_HR_KS_E02_D01_Z_Predcit_data.csv",header = T)
BCU40032 <- read.csv("./G01_bu_suan/G07_0718_high_Predcit_data_X_band.csv",header = T)
BCU400A <- BCU40032[,c(3:51)]
BCU400 <- BCU400A
dim(BCU400)
Classbcu <- BCU400[,43]




#BCU400 <- BCU400A[,-c(1,46)]
#dim(BCU400)
#names(BCU400)
#Y <- log10(BCU400[,c(64:70)])
#Classbcu <- BCU400[,46]
#names(Classbcu)
#names(Y)




TTime <- Sys.time();TTime
TTime1 <- format(TTime,"%Yy_%mm_%dd_%Hh_%Mm_%Ss")








xx <- c(1:23)








filename = paste("GHR04_high_DR2_4FGL_parallel_RF",TTime1,
                 "seed_123_gai",max(xx),"par_X_1_5.csv" 
                 ,sep = "_");filename



filename = "GHR04_high_DR2_4FGL_parallel_RF_2022y_07m_19d_19h_12m_11s_seed_123_gai_23_par_X_1_5.csv"


func_jk <- function(j){
    # #j=23
    # #print(j)
    # x1 <- zuhe[j,];x1
    # library(stringr)
    # x2 <- unlist(str_split(x1,","));  x2 # 以,进行分割
    # #x4 <- x2[[1]];x4
    # x3 <- as.numeric(x2);x3
    x3 <- comb_res[,j];x3



    
    #n_cal <- n_cal+1
    #cat(j,x3,"\n")
    RF_Accuracy0   <- 0.945
    #ANN_Accuracy0 <- 0.89
    #SVM_Accuracy0 <- 0.90
    
#RF
#x3 <- c(1,3,4,5,9,14,17,21,22,23)
#ANN
#x3 <- c(6,	8,	9,	13,	14,	19,	22)
#SVM
#x3 <- c(1,	2,	3,	4,	10, 13,	16,	17,	20,	21,	23)
#x3 <- c(1,3,4,5)    
###############################################################################
X <- (databf[,c(x3)]);
Y <- (BCU400[,c(x3)]);  
    
# X <- (databf[,c(x3)]);
# Y <- (BCU400[,c(x3)]);
##样本中随机去4/5的源，作为train set, 剩余1/5，为test set;
###############################################################################
set.seed(123)
train <- sample(1:nrow(X),size = round(nrow(X)*4/5),replace = FALSE)
X.train <- X[train,];  Class.train <- Class[train]
X.test  <- X[-train,]; Class.test  <- Class[-train]

# table(Class.train)
# table(Class.test)
###############################################################################
set.seed(123)
bcu_predict <- sample(1:nrow(Y),size = round(nrow(Y)*3/3), replace = FALSE)
Y.bcu_predict      <- Y[bcu_predict,]
Classu.bcu_predict <- Classbcu[bcu_predict]
###############################################################################
Class.train <- as.data.frame(Class.train)
colnames(Class.train) <- c("Optical.Class")
XX.train <- cbind(X.train,Class.train)
XX.train$Optical.Class = factor(XX.train$Optical.Class)
#第二种方法，Random Forest discriminate analysis

###############################################################################
library(randomForest)
XX.train <- na.roughfix(XX.train)
X.test <- na.roughfix(X.test)
Y.bcu_predict <- na.roughfix(Y.bcu_predict)
#set.seed(123)
RF_fit <- randomForest(Optical.Class ~.,  data=XX.train)
# model
#set.seed(123)
#RF_fit <- randomForest(Optical.Class ~.,  data=XX.train)
#RF_fit <- randomForest(Optical.Class ~.,  data=XX.train, na.action=na.omit)
##--------------------------------------------------------------
# test
RF_predct<- predict(RF_fit,newdata=X.test)
RF_perf <- table(Class.test, RF_predct,dnn=c("Actual", "Predicted"))
# predict
RF_predctbcu <- predict(RF_fit,newdata=Y.bcu_predict)
##--------------------------------------------------------------
###############################################################################
###############################################################################
###############################################################################
X <- (databf[,c(x3)]);
Y <- (BCU400[,c(x3)]);
##样本中随机去4/5的源，作为train set, 剩余1/5，为test set;
###############################################################################
set.seed(123)
train <- sample(1:nrow(X),size = round(nrow(X)*4/5),replace = FALSE)
X.train <- X[train,];  Class.train <- Class[train]
X.test  <- X[-train,]; Class.test  <- Class[-train]
###############################################################################
set.seed(123)
bcu_predict <- sample(1:nrow(Y),size = round(nrow(Y)*3/3), replace = FALSE)
Y.bcu_predict      <- Y[bcu_predict,]
Classu.bcu_predict <- Classbcu[bcu_predict]
###############################################################################
Class.train <- as.data.frame(Class.train)
colnames(Class.train) <- c("Optical.Class")
XX.train <- cbind(X.train,Class.train)
#第二种方法，Random Forest discriminate analysis
###############################################################################
# library(nnet)
# ANN_fit <-  nnet(Optical.Class~.,data=XX.train,size=2,trace=FALSE)
# ANN_predct  <-  predict(ANN_fit,X.test,type="class")
# ANN_predctbcu <- predict(ANN_fit,Y.bcu_predict,type="class")
# ANN_perf <- table(Class.test, ANN_predct, dnn=c("Actual", "Predicted"))
# ##############################################################################
# ##############################################################################
# ##############################################################################
# library(e1071)
# SVM_fit <- svm(Optical.Class~.,data=XX.train)
# SVM_predct <- predict(SVM_fit, na.omit(X.test))
# SVM_predctbcu <- predict(SVM_fit, na.omit(Y.bcu_predict))
# SVM_perf <- table(Class.test, SVM_predct,dnn=c("Actual", "Predicted"))
###############################################################################
library(e1071)
#Random Forest
RF_Accuracy <- classAgreement(RF_perf)
RF_number   <- table(RF_predctbcu)
RF_number   <- as.data.frame(RF_number)
RF_data <- c(RF_number[1,2],RF_number[2,2],RF_Accuracy[["diag"]])
RF_data


# # SVM
# SVM_Accuracy <- classAgreement(SVM_perf)
# SVM_number   <- table(SVM_predctbcu)
# SVM_number   <- as.data.frame(SVM_number)
# SVM_data <- c(SVM_number[1,2],SVM_number[2,2],SVM_Accuracy[["diag"]])
# SVM_data
# 
# # ANN
# ANN_Accuracy <- classAgreement(ANN_perf)
# ANN_number   <- table(ANN_predctbcu)
# ANN_number   <- as.data.frame(ANN_number)
# ANN_data <- c(ANN_number[1,2],ANN_number[2,2],ANN_Accuracy[["diag"]])
# ANN_data

#print(classAgreement(RF_perf)$diag); print(table(RF_predctbcu))
#print(classAgreement(ANN_perf)$diag); print(table(ANN_predctbcu))
#print(classAgreement(SVM_perf)$diag); print(table(SVM_predctbcu))


#某组参数的总汇
RF_result_data1 <-   c(length(x3),RF_data, x3)
as.data.frame(RF_result_data1)
RF_result_data2 <- t(data.frame(RF_result_data1))

# ANN_result_data1 <-   c(length(x3),ANN_data, x3)
# as.data.frame(ANN_result_data1)
# ANN_result_data2 <- t(data.frame(ANN_result_data1))
# 
# SVM_result_data1 <-   c(length(x3),SVM_data, x3)
# as.data.frame(SVM_result_data1)
# SVM_result_data2 <- t(data.frame(SVM_result_data1))






if(RF_Accuracy[["diag"]] >= RF_Accuracy0) {
  RF_Accuracy0 <- RF_Accuracy[["diag"]]
  write.table(RF_result_data2,
              #file = "ML4FGL_dr2_parallel_v4_RF_20201030_seed_123_gai.csv",
              file = filename,
              sep = ",",col.names = FALSE,append = TRUE )}


# if(ANN_Accuracy[["diag"]] >= ANN_Accuracy0) {
#   ANN_Accuracy0 <- ANN_Accuracy[["diag"]]
#   write.table(ANN_result_data2,
#               file = "ML4FGL_parallel_v4_ANN_20191003_test.csv",
#               sep = ",",col.names = FALSE,append = TRUE )}
# 
# if(SVM_Accuracy[["diag"]] >= SVM_Accuracy0) {
#   SVM_Accuracy0 <- SVM_Accuracy[["diag"]]
#   write.table(SVM_result_data2,
#               file = "ML4FGL_parallel_v4_SVM_20191003_test.csv",
#               sep = ",",col.names = FALSE,append = TRUE )}
return(j+1)
}




func_total_N <- function(j){
  xx3 <- c(1:j)
  max_BB <- matrix(0,j)
  for (i in 1:j){
    max_BB[i] <-  max(choose(xx3,i))
  }
  sum_max_B <- sum(max_BB)
  return(sum_max_B)
}










print('#######################################################################')
print('#                                                                     #')
print('#######################################################################')
classiftime0 <- proc.time() # record classification time 



mx <- max(xx);mx
mxxc <- c(2,22,23)

#for (l in 2:mx) {
#for (l in 2:5) {
for (l in mxxc) {
    
  print(paste("A total of",mx,"parameters"))
  print(paste("Number",l,"of", mx))
  comb_res <- combn(xx,l);comb_res
  min_A <- 1
  max_A <- ncol(comb_res); max_A #计算组合所在列数，即组合个数
  
  print(paste("Numbers of the cycle is", max_A)) 

  
  
  
  print('########################################')
  print('########################################')
  print('########################################')
  classiftime1 <- proc.time() # record classification time
  # # 用 snowfall 并行计算
  print('用 snowfall 并行计算:')
  #加载parallel包
  library(parallel)  #用于并行计算
  library(snowfall)  # 载入snowfall包,用于并行计算
  #detectCores函数可以告诉你你的CPU可使用的核数
  #clnum<-detectCores(logical = F) - 1
  clnum<-detectCores()
  #print('The core of the computer is:'); print(clnum)
  print(paste("The core of the computer is :", clnum, "cores"))
  # 并行初始化
  sfInit(parallel = TRUE, cpus = clnum, slaveOutfile = "Routtest.txt")

  # sfLibrary(randomForest)
  # sfLibrary(e1071)
  sfExport("comb_res")
  sfExport("databf","BCU400","Class","Classbcu")
  sfExport("filename")
  sfLibrary(parallel)
  sfLibrary(snowfall)
  
  print(paste("Number",l,"of", mx, "is running......"))  
  # 进行lapply的并行操作
  s<-sfLapply(min_A:max_A, func_jk)  
  # 结束并行，返还内存等资源
  sfStop()

  classiftime <- proc.time() - classiftime1
  print(classiftime)
  print(classiftime[3])

  print('########################################')
  print('########################################')
  print('########################################')

  

  
  
  
  
  
  
  
  
  
  
  
  
  
  

# print('########################################')
# print('########################################')
# print('########################################')
# classiftime1 <- proc.time() # record classification time
# 
# # # 用 mclapply 并行计算
# print('用 mclapply 并行计算:')
# #加载parallel包
# library(parallel)
# #detectCores函数可以告诉你你的CPU可使用的核数
# clnum<-detectCores()
# print('The core of the computer is:'); print(clnum)
# mc <- getOption("mc.cores", clnum)
# res <- mclapply(min_A:max_A, func_jk, mc.cores = mc)
# 
# classiftime <- proc.time() - classiftime1
# print(classiftime)
# print(classiftime[3])
# 
# print('########################################')
# print('########################################')
# print('########################################')







# #print('########################################')
# #print('########################################')
# print('########################################')
# classiftime1 <- proc.time() # record classification time
# #
# # 用 parLapply 并行计算
# print('用 parLapply 并行计算:')
# #加载parallel包
# library(parallel)
# #detectCores函数可以告诉你你的CPU可使用的核数
# clnum<-detectCores()
# #print('The core of the computer is '); print(clnum)
# print(paste("The core of the computer is :", clnum, "cores"))
# print(paste("Number",l,"of", mx, "is running......"))
# 
# x <- c(min_A:max_A)
# mc <- getOption("cl.cores", clnum)
# cl <- makeCluster(mc)
# ## to make this reproducible
# #clusterSetRNGStream(cl, 123)
# 
# varlist <- c("max_A","min_A","databf","BCU400",
#              "Class","Classbcu","filename",
#              "comb_res")
# 
# # varlist <- c("zuhe","max_A","min_A","databf","BCU400",
# #              "Class","Classbcu","filename")
# 
# clusterExport(cl = cl, varlist, envir = .GlobalEnv)
# xxx <- do.call(c, parLapply(cl, x, func_jk))
# stopCluster(cl)
# 
# 
# classiftime <- proc.time() - classiftime1
# print(classiftime)
# print(classiftime[3])
# print('########################################')
# #print('########################################')
# #print('########################################')

  




# print('########################################')
# print('########################################')
# print('########################################')
# classiftime1 <- proc.time() # record classification time
# 
# # 用 foreach 并行计算
# print('用 foreach 并行计算:')
# library(parallel)
# # Calculate the number of cores
# no_cores <- detectCores();
# #no_cores <-24
# print('The core of the computer is '); print(no_cores)
# # Initiate cluster
# #cl <- makeCluster(no_cores)
# # 启用parallel作为foreach并行计算的后???
# #library(help="doParallel")
# library(doParallel)
# 
# 
# cl <- makeCluster(no_cores)
# registerDoParallel(cl)
#   # 并行计算方式
# x <- foreach(j=min_A:max_A,.combine='rbind')%dopar%func_jk(j)
# stopCluster(cl)
# 
# 
# classiftime <- proc.time() - classiftime1
# print(classiftime)
# print(classiftime[3])
# print('########################################')
# print('########################################')
# print('########################################')
# print("Average number of calculations per second")


#print(max_A/classiftime[3])
Num_s <- max_A/classiftime[3]

print(paste("，，，，，，，，，，，，，，，，，", classiftime[3]/3600," 小时 "))
print(paste("，，，，，，，，，，，，，，，，，", classiftime[3]/3600," 小时 "))

print('########################################')
print(paste("Number of calculations per second is", max_A/classiftime[3]," 次"))
print(paste("Number of calculations per second is", max_A/classiftime[3]," 次"))

print(paste(26,"paramters need:",func_total_N(26)/(Num_s*24*60*60),"days"))
print(paste(25,"paramters need:",func_total_N(25)/(Num_s*24*60*60),"days"))
print(paste(24,"paramters need:",func_total_N(24)/(Num_s*24*60*60),"days"))
print(paste(23,"paramters need:",func_total_N(23)/(Num_s*24*60*60),"days"))
print(paste(22,"paramters need:",func_total_N(22)/(Num_s*24*60*60),"days"))
print(paste(21,"paramters need:",func_total_N(21)/(Num_s*24*60*60),"days"))
print(paste(20,"paramters need:",func_total_N(20)/(Num_s*24*60*60),"days"))




print('########################################')
print('########################################')
print('########################################')
print('#######################################################################')
print('#######################################################################')
 

# print("Approximately days 26 paramters")
# print(67108863/(Num_s*24*60*60))
# print("Approximately hours 26 paramters")
# print(67108863/(Num_s*60*60))
# 
# 
# 
# print("Approximately days 23 paramters")
# print(8388607/(Num_s*24*60*60))
# print("Approximately hours 23 paramters")
# print(8388607/(Num_s*60*60))
#
# print(func_total_N(26)/(Num_s*24*60*60))
# print(func_total_N(25)/(Num_s*24*60*60))
# print(func_total_N(24)/(Num_s*24*60*60))
# print(func_total_N(23)/(Num_s*24*60*60))
# print(func_total_N(22)/(Num_s*24*60*60))
# print(func_total_N(21)/(Num_s*24*60*60))


#classiftime0 <- proc.time() # record classification time
Alltime <- proc.time() - classiftime0
print("The total running time of the program")
print(paste("The total running time  is", Alltime[3],"seconds"))
print(paste("The total running time  is", Alltime[3]/(60*60),"hours"))
print(paste("The total running time  is", Alltime[3]/(60*60*24),"days"))
print('#######################################################################')
print('#                                                                     #')
print('#######################################################################')
print(paste("#                  ", date(),       "                         #'"))
print('#######################################################################')
print('#                                                                     #')
print('#######################################################################')


} #emd mx


Alltime <- proc.time() - classiftime0
print(paste("The total running time  is", Alltime[3],"seconds"))
print(paste("The total running time  is", Alltime[3]/(60*60),"hours"))
print(paste("The total running time  is", Alltime[3]/(60*60*24),"days"))
print('#######################################################################')

### 
### over code
###


