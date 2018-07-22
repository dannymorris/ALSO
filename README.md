---
title: "Attribute-Wise Learning for Scoring OUtliers"
author: "Danny Morris"
date: "July 22, 2018"
output:
    html_document:
        keep_md: true                           
---


# Attribute-Wise Learning for Scoring Outliers (ALSO)

*Attribute-wise Learning for Scoring Outliers* (ALSO) is an unsupervised anomaly detection algorithm for multidimensional data. The main goal is to automate separate predictive models for each feature in the dataset. In each model, one feature in the dataset is the target and the remaining are predictors. A classifier or regressor is called to model the relationships. Observations are scored and compared to true values, and a numerical outlier score is returned for each observation based on the magnitude of deviation from expected values. Given the vector of numeric outlier scores, simple univariate techniques can be used to extract insights. Larger outlier scores suggest greater outlierness.

# Install


```r
devtools::install_github("dannymorris/ALSO")
```

# Example using Random Forest

In *Outlier Analysis* (C.C Aggarwal. Springer, 2017), the author recommends the use of random forests as the base regressor. Random forests are highly effective in both prediction and classification settings due to their robustness.

Here we'll examine the `datasets::state.x77` dataset. It contains 50 observations and 8 variables. The dataset is rather small, but it works well for illustration.


```r
library(dplyr)
```

```
## Warning: package 'dplyr' was built under R version 3.4.4
```

```
## 
## Attaching package: 'dplyr'
```

```
## The following objects are masked from 'package:stats':
## 
##     filter, lag
```

```
## The following objects are masked from 'package:base':
## 
##     intersect, setdiff, setequal, union
```

```r
library(ALSO)
```

#### Quick data prep

Currently only data frame or tibbles are supported as inputs to the function.


```r
# z-standardize all columns
zstd <- function(x) {
    (x - mean(x)) / sd(x)
}


state_tbl <- state.x77 %>%
    dplyr::as_tibble() %>%
    dplyr::mutate_all(zstd)
```

```
## Warning: package 'bindrcpp' was built under R version 3.4.4
```

#### Using the ALSO() function


```r
rf_also <- ALSO::ALSO(data = state_tbl,
                      model_function = randomForest::randomForest,
                      cross_validate = TRUE,
                      n_folds = 5,
                      scores_only = TRUE)

rf_also
```

```
##  [1] 0.42872865 4.08904263 1.61786642 0.91365745 2.98547371 1.21455210
##  [7] 0.72866829 0.59627654 0.72534875 0.78233355 4.65714841 0.18186000
## [13] 0.58818213 0.15124479 0.19475919 0.13029883 0.30309778 0.83754926
## [19] 1.04725939 0.59718600 0.52987254 0.37122491 0.42074598 0.43994690
## [25] 0.53504337 0.62550781 0.09051616 3.18581242 0.20364873 0.47772908
## [31] 1.66615719 0.79468721 0.36532007 2.14265338 0.22773227 0.47532714
## [37] 0.73778970 0.64915310 1.50566987 0.80441917 0.58256787 0.06425327
## [43] 1.42040498 1.08427293 0.55749495 0.15719753 1.18108084 0.98060295
## [49] 0.18809725 0.55823879
```

Here we return only the outlier scores. Observations are given out-of-sample scores via 5-fold cross validation.


```r
lm_also <- ALSO::ALSO(data = state_tbl,
                      model_function = lm,
                      cross_validate = TRUE,
                      n_folds = 5,
                      scores_only = FALSE)

lm_also
```

```
## $scores
##  [1] 0.54460740 6.31540090 0.83172884 0.36672603 0.29688862 0.70372163
##  [7] 0.56746221 0.25611797 0.65903073 0.08251311 2.83278938 0.17837391
## [13] 0.08027064 0.12467682 0.08689756 0.17480662 0.62181536 0.72909820
## [19] 1.03238576 0.25013314 0.62635091 0.49705084 0.20598721 0.35609812
## [25] 0.71080029 0.43840737 0.01587065 1.40448094 0.35720090 0.27606963
## [31] 2.42104222 0.35241077 0.13350536 0.84462249 0.02997737 0.13389983
## [37] 2.16063688 0.48838327 0.43200535 0.30277904 0.19372300 0.22211869
## [43] 0.39849400 1.02217122 0.68191563 0.05336092 2.47352449 0.61308419
## [49] 0.12589817 0.30405897
## 
## $squared_prediction_errors
## # A tibble: 50 x 8
##    Population   Income Illiteracy `Life Exp`  Murder `HS Grad`   Frost
##         <dbl>    <dbl>      <dbl>      <dbl>   <dbl>     <dbl>   <dbl>
##  1   0.495     0.0860     0.159    0.428     0.851     0.184   0.147  
##  2  15.8      31.5        0.0860   6.28      6.44     12.0     0.936  
##  3   0.0208    0.00135    0.209    0.483     0.357     0.592   1.35   
##  4   0.514     0.425      0.0217   0.610     0.201     0.681   0.00257
##  5  11.4       2.03       0.0129   0.00921   0.132     0.641   0.463  
##  6   0.295     0.0199     0.0520   0.578     0.479     0.199   1.13   
##  7   0.000116  3.15       0.853    0.121     0.00953   0.00344 0.612  
##  8   0.534     0.204      0.00757  0.340     0.0818    0.00962 0.448  
##  9   0.181     0.552      0.483    0.0000327 0.155     0.0395  1.39   
## 10   0.214     0.213      0.0352   0.00196   0.0697    0.235   0.0105 
## # ... with 40 more rows, and 1 more variable: Area <dbl>
## 
## $adjusted_feature_weights
## Population     Income Illiteracy   Life Exp     Murder    HS Grad 
##  0.0000000  0.0000000  0.4146261  0.2800226  0.3345996  0.1745297 
##      Frost       Area 
##  0.2872309  0.0000000
```

Here we use Ordinary Least Squares regression as the base regressor, 5-fold cross validation for scoring, and a list containg various artifacts from the algorithm:

