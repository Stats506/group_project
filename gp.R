library(foreign)
library(tidyverse)
setwd("/Users/Amy/Desktop/Stats506_F19/GroupProject")

# data preparation
hsq = read.xport("HSQ_D.XPT")
alq = read.xport("ALQ_D.XPT")
demo = read.xport("DEMO_D.XPT")

health = hsq %>%
  filter(HSD010 <= 3) %>%
  transmute(id = SEQN, health = HSD010)

drinks = alq %>%
  filter(ALQ120Q <= 365) %>%
  transmute(id = SEQN, drink = ALQ120Q)

demo = demo %>%
  filter(RIDAGEYR >= 21) %>%
  transmute(id = SEQN, sex = RIAGENDR, age = RIDAGEYR, 
            pir = INDFMPIR, edu = DMDEDUC2)

data = drinks %>%
  left_join(health, by = "id") %>%
  left_join(demo, by = "id") %>%
  na.omit()

# linear regression model
mod = lm(drink ~. - id, data = data)

# model output
summary(mod)
plot(mod$fitted.values, mod$residuals)
par(mfrow = c(2, 2))
plot(mod)
par(mfrow = c(1, 1))

# resample residuals with replacement
sample_residuals = function(orig_x, model) {
  # input: 
  #   orig_x - a data frame of predictor values from the original linear model
  #   model  - a linear regression model
  # output: 
  #   a data frame representing with new responses for linear regression model
  #   after resampling residuals
  e = sample(residuals(model), size = length(residuals(model)), replace = TRUE)
  new_y = fitted(model) + e
  df = cbind(new_y, orig_x)
  return(df)
}
# test function
new_data = sample_residuals(orig_x, mod)

# Estimators after resampling residuals
get_coeff = function(data) {
  # inputs: data - a data frame
  # output: vector of linear regression coefficients
  model = lm(new_y ~., data = data)
  return(coefficients(model))
}
# test function
get_coeff(new_data)

# Confidence intervals by resampling residuals
boot_ci = function(B, alpha) {
  # inputs:
  #   B - a positive interger, number of bootstrap replicates
  #   alpha - alpha level for confidence interval
  # outpus:
  #   array of upper and lower confidence limits for each coefficient
  
  nboot = replicate(B, get_coeff(sample_residuals(orig_x, mod)))
  low = apply(nboot, 1, quantile, probs = alpha/2)
  up = apply(nboot, 1, quantile, probs = 1 - alpha/2)
  lwr = 2*coefficients(mod) - up
  upr = 2*coefficients(mod) - lwr
  ci = rbind(lwr, est = coefficients(mod), upr)
  return(ci)
}

# test function
boot_ci(10, .05)



