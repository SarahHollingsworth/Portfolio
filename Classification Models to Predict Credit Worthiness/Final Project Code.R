credit_df <- read.csv("C:\\Users\\rewar\\OneDrive\\Documents\\DATA 621\\Final Project\\german_credit.csv", header=TRUE)

Credit = as.factor(ifelse(credit_df$V21==1, "Good", "Bad"))
credit_df$V21 <- NULL
credit_df <- data.frame(credit_df, Credit)
colnames(credit_df) <- c("Index", "Stat_Ex_Chkg", "Dur_in_Mon", "Cr_Hist", "Purpose", "Cr_Amt", "Sav_Acc_Bd", "Emp_Yrs", "Rate_Per_Dis_Inc", "Sex_Status", "Oth_Debtors", "Pres_Res_Since", "Property", "Age_Yrs", "Other_Plans", "Housing", "No_Exst_Cred", "Job", "Dependents", "Telephone", "Foreign_Wkr", "Credit")
str(credit_df)


library(ggplot2)
library(tidyverse)
ggplot(data = credit_df) +
  geom_bar(mapping = aes(x = Sex_Status, fill = Credit))

ggplot(data = credit_df) +
  geom_bar(mapping = aes(x = Stat_Ex_Chkg, fill = Credit))

ggplot(data = credit_df) +
  geom_bar(mapping = aes(x=Cr_Hist, fill=Credit))

ggplot(data = credit_df) +
  geom_point(mapping = aes(x=Dur_in_Mon, y=Cr_Amt, color=Credit))

ggplot(data = credit_df) +
  geom_point(mapping = aes(x = Age_Yrs, y=Cr_Amt, color=Credit))

ggplot(data = credit_df) +
  geom_bar(mapping = aes(x = Purpose, fill = Credit))

ggplot(data = credit_df) +
  geom_point(mapping = aes(x = Cr_Amt, y=Stat_Ex_Chkg, color=Credit))

ggplot(data = credit_df) +
  geom_point(mapping = aes(x=Purpose, y = Dur_in_Mon, color=Credit))

ggplot(data = credit_df) +
  geom_point(mapping = aes(x=Age_Yrs, y = Dur_in_Mon, color=Credit))

ggplot(data = credit_df) +
  geom_bar(mapping = aes(x=Dur_in_Mon, fill=Credit))

ggplot(data = credit_df) +
  geom_bar(mapping = aes(x=Age_Yrs, fill=Credit))

ggplot(data = credit_df) +
  geom_point(mapping = aes(x=Cr_Amt, y=Age_Yrs, color=Credit))

ggplot(data = credit_df) +
  geom_point(mapping = aes(x=Pres_Res_Since, y=Cr_Amt, color=Credit))

ggplot(data = credit_df) +
  geom_point(mapping = aes(x=No_Exst_Cred, y=Age_Yrs, color=Credit))

ggplot(data = credit_df) +
  geom_bar(mapping = aes(x=Purpose, fill=Purpose))

ggplot(data = credit_df) +
  geom_bar(mapping = aes(x=Purpose, fill=Credit))

ggplot(data = credit_df) +
  geom_bar(mapping = aes(x=Credit, fill=Credit))



cols <- c("Stat_Ex_Chkg", "Cr_Hist", "Purpose", "Sav_Acc_Bd", "Emp_Yrs", "Sex_Status", "Oth_Debtors", "Property", "Other_Plans", "Housing", "Job", "Telephone", "Foreign_Wkr")
credit_df[cols] <- lapply(credit_df[cols], factor)
str(credit_df)

detach(credit_df)
attach(credit_df)
library(randomForest)
set.seed(1)
train = sample(1:nrow(credit_df), 0.7 * nrow(credit_df))
credit_df_test = credit_df[-train,]
Credit_test = credit_df$Credit[-train]
rf_df = randomForest(Credit~.-Index, data=credit_df, subset=train, mtry=6, importance=TRUE)
plot(rf_df)
varImpPlot(rf_df, cex=0.7)
rf_df$confusion
yhat_rf_df = predict(rf_df, newdata=credit_df[-train,])
table(predict=yhat_rf_df, truth=Credit_test)


credit_df$Credit = as.numeric(ifelse(credit_df$Credit=="Good", 0, 1))
library(gbm)
boost_credit_df = gbm(Credit~.-Index, data=credit_df[train,], distribution="bernoulli", 
                      n.trees=2000, interaction.depth=4)
summary(boost_credit_df)

credit_df$Credit <- as.factor(ifelse(credit_df$Credit==0, "Good", "Bad"))
library(tree)
tree_df <- tree(Credit~Purpose+Cr_Amt+Stat_Ex_Chkg+Age_Yrs+Dur_in_Mon+Cr_Hist, credit_df, subset=train)
summary(tree_df)
plot(tree_df)
text(tree_df, pretty=0, cex=0.5)



cv_tree = cv.tree(tree_df, FUN=prune.misclass)
cv_tree
par(mfrow=c(1,2))
plot(cv_tree$size, cv_tree$dev, type="b")
plot(cv_tree$k, cv_tree$dev, type="b")

par(mfrow=c(1,1))
prune_credit_df = prune.misclass(tree_df, best=4)
plot(prune_credit_df)
text(prune_credit_df, pretty=0, cex=0.5)

yhat_prune = predict(prune_credit_df, newdata=credit_df[-train, ])
dim(y)
plot(yhat_prune, Credit_test)
abline(0, 1)
mean((yhat_prune-Credit_test)^2)

library(e1071)
set.seed(1)
svm_fit = svm(Credit~.-Index, data=credit_df[train,], kernel="radial", gamma=1, cost=0.1, scale=FALSE)
summary(svm_fit)


tune_svm = tune(svm, Credit~.-Index, data=credit_df[train,], kernel="radial", 
                ranges=list(cost=c(0.1, 1, 10, 100, 1000),
                gamma=c(0.5, 1, 2, 3, 4)))
summary(tune.out)

best_model = tune_svm$best.model
summary(best_model)

train = sample(1:nrow(df), 0.7 * nrow(df))
df_test = df[-train,]
Credit_test = df$Credit[-train]

y_pred = predict(best_model, df_test)
table(predict=y_pred, truth=Credit_test)
(29 + 195) / 300

credit_df$Credit <- as.factor(ifelse(credit_df$Credit==0, "Good", "Bad"))


pr_comp_credit = prcomp(credit_df, scale=TRUE)
biplot(pr_comp_credit, scale=0)
pr_var = pr_comp_credit$sdev^2
pve = pr_var/sum(pr_var)
plot(pve, xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1), type="b")
