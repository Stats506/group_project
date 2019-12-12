## 506 Group Project
#
# How are drinking habits related to general health condition for adults in the USA ?
#
# Fit the linear regression of response frequency of drinking alcohol in the past
# 12 months with the predictors health condition, age, gender and, PIR
#
# Samples were limited to helathy people aged over 21 years (3 healthy levels)
# continuous variables frequency of drinking, age and PIR was standardized
# 
# Author: Ningyuan Wang
# Date: December 1, 2019
#
# Data manipulation was done with package dplyr() in R


# useful links
# https://www.dummies.com/education/science/biology/the-bootstrap-method-for-standard-errors-and-confidence-intervals/
# above link is helpful for understanding bootstrap in estimating se and ci

# Method: bootstrapping residuals
#------------------------------------------------------------------------------
## libraries
library(Hmisc) # for read xpt datasets 
library(dplyr)
library(tidyverse)
library(boot)
library(ggplot2)

# read in the datasets and select the variables 
alcohol = sasxport.get("https://wwwn.cdc.gov/Nchs/Nhanes/2005-2006/ALQ_D.XPT") %>% 
  select(seqn, alq120q) %>%
  filter(alq120q < 366) %>%
  drop_na()

health = sasxport.get("https://wwwn.cdc.gov/Nchs/Nhanes/2005-2006/HSQ_D.XPT") %>%
  select(seqn, hsd010) %>%
  filter(hsd010 <=3 ) %>%
  drop_na() # limit to the healthy samples


demography = sasxport.get("https://wwwn.cdc.gov/Nchs/Nhanes/2005-2006/DEMO_D.XPT") %>%
  filter(ridageyr >=21,dmdeduc2 <= 5 ) %>%
  transmute(seqn, age = ridageyr, gender = riagendr, education = dmdeduc2, pir = indfmpir ) %>%
  drop_na() # limit to people > 20 years
  


# join above datasets
df = left_join(alcohol, health, by = "seqn") %>% 
  left_join(., demography, by = "seqn") %>%
  drop_na() %>%
  rename(drinks = alq120q, health = hsd010) # 2791*7

# convert variables to factors and rescale continous variables
df = df %>% 
  mutate_at(vars(seqn, health, gender, education), as.factor) %>%
  mutate(drinks_std = scale(drinks),
         age_std = scale(age),
         pir_std = scale(pir))




#  plot the relation and fit the model
ggplot(df, aes(x=health, y= drinks_std)) + 
  geom_point(color='#2980B9', size = 4) + 
  geom_smooth(method=lm, color='#2C3E50')



lm1 = lm(drinks_std ~ health + age_std + gender + 
           education + pir_std, data = df) 

summary(lm1) # only gender and edcuation are statistically significant, RSE = 0.999 and R^2 = 0.0053


# model diagnostics 
res = lm1$residuals 
fit = lm1$fitted
plot(fit, res, xlab = "fitted", ylab = "Residuals") # unequal varlaince in residuals
qqnorm(lm1$residuals, ylab = "Residuals"); qqline(lm1$residuals) # non-normal residual

# 也许没用： 了解residual, not normal 
mean(res) #-2.097341e-18
median(res) #-0.1169909


# compute the coverage probability of response , 这一部分非常不确定, 从这开始好好想想
t = qt(1 - 0.05/2, df = lm1$df.residual)
rse = sqrt(sum(res^2) / lm1$df.residual) # 0.9997812
lwr = fit - t * rse
upr = fit + t * rse


ci_df = cbind(df$drinks_std, lwr, upr) %>%
  as_tibble() %>%
  transmute(obs = V1, lwr = lwr, upr = upr, 
            cover = as.numeric(obs > lwr & obs < upr) )

coverage = sum(ci_df$cover) / nrow(ci_df) # 0.9914009
length = mean(ci_df$upr - ci_df$lwr) # 3.920777
  


# bootstrapping regression residuals

# a. construct bootstrap samples : res and new response y*
n = nrow(df) # sample size
B = 1e3  # number of bootstrap samples 

boot_samples = sample(res, n * B, replace = TRUE)
dim(boot_samples) = c(n, B) # each column is a dataset
boot_y = matrix(fit, n, ncol(boot_samples)) + boot_samples # bootstrap y as faked response

# with the graphs, we see the response becomes normal: may be deleted
hist(df$drinks_std) # orginal response
hist(rowMeans(boot_y)) # bootstrap response (faked response)
quantile(rowMeans(boot_y)) # 95% quantile CI for bootstrap y 



# b. refit the model with bootstrap samples on y* and get the estimate of coefficents
lm_coef = function(y) lm(y ~ df$health + df$age_std + df$gender + df$education + df$pir_std)$coef
boot_lms = apply(boot_y, 2, lm_coef)



# c. get the bootstrap estimates of the model: why choose mean or median 
boot_mean_est = rowMeans(boot_lms) # mean estimate 
boot_median_est = apply(boot_lms, 1, median) # median estimate 

# d. estimates covariance and standard deviation

b_se = boot_se(boot_y, lm_fun)
b_se$cov

b_se$theta

hist(b_se$theta[1, ], main = expression(hat(b)[0]))
hist(b_se$theta[2, ], main = expression(hat(b)[1]))
hist(b_se$theta[3, ], main = expression(hat(b)[2]))


boot_se = function(boot_y, fun, ...) {
  if(is.matrix(boot_y)){ # get coefficent estimates
    theta = apply(boot_y, 2, fun, ...)
  } else {
    theta = apply(boot_y, 3, fun, ...)
  }
  
  if(is.matrix(theta)){ # covariances and standard deviation of each estimates
    return(list(theta = theta, cov = cov(t(theta))))
  } else{
    return(list(theta = theta, se = sd(theta)))
  }
}

# e. compute the coverage of the response

# e1. confidence interval with mean 
lm_res = function(y) lm(y ~ df$health + df$age_std + df$gender + df$education + df$pir_std)$residuals
boot_res = apply(boot_y, 2, lm_res)
boot_rse = sqrt(colSums(boot_res^2) / lm1$df.residual)

lm_fit = function(y) lm(y ~ df$health + df$age_std + df$gender + df$education + df$pir_std)$fitted.values
boot_fit = apply(boot_y, 2, lm_fit)


boot_rse  = matrix(boot_rse, nrow = n, ncol = B, byrow = TRUE) # expand boot_rse to a matrix


boot_lwr = boot_fit - t * boot_rse
boot_upr = boot_fit + t * boot_rse

coverage_m = length(which((boot_y > boot_lwr & boot_y < boot_upr) )) / (n * B) #0.9912307
length_m = mean(boot_upr - boot_lwr ) # 3.825179



# e2. confidence interval with quantile (median)
boot_quan = quantile(as.vector(boot_fit), probs = c(.025,.975)) # (-0.177652  0.162630 )
ci_q = as.vector(boot_y) %>% 
  as_tibble() %>%
  mutate(lwr_q = boot_quan[1], upr_q = boot_quan[2], cover = as.numeric(value > lwr_q & value < upr_q))

coverage_q = sum(ci_q$cover) / nrow(ci_q) # 0.6261662
length_q = mean(ci_q$upr_q - ci_q$lwr_q) # 0.340282




boot_se = function(boot_y, fun, ...) {
  if(is.matrix(boot_y)){ # get coefficent estimates
    theta = apply(boot_y, 2, fun, ...)
  } else {
    theta = apply(boot_y, 3, fun, ...)
  }
  
  if(is.matrix(theta)){ # covariances and standard deviation of each estimates
    return(list(theta = theta, cov = cov(t(theta))))
  } else{
    return(list(theta = theta, se = sd(theta)))
  }
}


b_se = boot_se(boot_y, lm_fun)
b_se$cov

b_se$theta


# estimate
mean_boot = colMeans(boot_samples) # 1000个mean
sd(mean_boot) #0.01857882
median_boot = apply(boot_samples, 2, median) # 1000个median
sd(median_boot) #0.002904569

# for the non-normality distributed data, we think median is better than mean estimate


# for each bootstrap resample
rse_boot = sqrt(colSums(boot_samples ^2) / lm1$df.residual) # 1000 rse
fit_mat = matrix(fit * B, nrow = n, ncol = B)
lwr_boot = fit_mat - t * rse_boot 
upr_boot = fit_mat + t * rse_boot



ci_boot = cbind(df$drinks_std, lwr_boot, upr_boot) %>%
  as_tibble() %>%
  transmute(obs = V1, lwr = lwr_boot, upr = upr_boot, 
            cover = as.numeric(obs > lwr_boot & obs < upr_boot) )

coverage = sum(ci_boot$cover) / nrow(ci_df) # 0.9921175
length = mean(ci_boot$upr - ci_boot$lwr) # 3.920777

# compute grant mean of bootstrap samples 
res_grand = mean(boot_samples) # 0.0005153294
# res_med = median(boot_samples) # -0.11701806
rse_boot =  sqrt( colSums(boot_samples^2) / lm1$df.residual)
rse_grand = mean(rse_boot) #0.9945817

# bootstrap estimates
est_mean = res_grand + y_bar # 0.0005153294
# est_med = res_med + y_bar #  -0.1170181


# confidence interval 
lwr_boot = est_mean - t * rse_grand
upr_boot = est_mean + t * rse_grand

cover_boot = cbind(df$drinks_std, lwr_boot, upr_boot) %>%
  as_tibble() %>%
  transmute(obs = V1, lwr_boot = lwr_boot, upr_boot = upr_boot, 
            cover_boot = as.numeric(obs >= lwr_boot & obs<= upr_boot) )


coverage_boot = sum(cover_boot$cover_boot) / nrow(ci_df) # 0.9921175
length_boot = mean(upr_boot - lwr_boot) #3.900381

###----------------------------------------------------------------------------



fit_b = colMeans(boot_samples) # mean for each bootstrap sample

rse_b = sqrt(sum((rowMeans(boot_samples))^2)) / lm1$df.residual # 0.0005847888, much smaller

# here we use percentile method to compute confidence interval 
alpha = 0.05
percent_ci = quantile(fit_b, c(alpha/2, 1-alpha/2)) 

boot_ci = c(lwr = 2 * fit_grand - unname(percent_ci[2]), 
           upr = 2 * fit_grand - unname(percent_ci[1] ))

cover_df = cbind(df$drinks, percent_ci[1], percent_ci[2], boot_ci[1], boot_ci[2] ) %>%
  as_tibble() %>%
  transmute(obs = V1, lwr_p = V2, upr_p = V3, lwr_b = V4, 
            upr_b = V5,
            cover_p = as.numeric(obs >= lwr_p & obs<= upr_p),
            cover_b = as.numeric(obs >= lwr_b & obs<= upr_b))

coverage_p = sum(cover_df$cover_p) / nrow(ci_df) # 0.2153076
length_p = mean(cover_df$upr_p - cover_df$lwr_p)  # 0.8545129

coverage_b = sum(cover_df$cover_b) / nrow(ci_df) # 0.2153076
length_b = mean(cover_df$upr_b - cover_df$lwr_b)  # 0.8545129






