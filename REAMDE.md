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
##  [1] 0.43579665 4.71543397 1.38120901 0.98941954 2.96756030 1.06870582
##  [7] 0.58732495 0.67896703 0.59878727 0.83965206 3.92845350 0.15874104
## [13] 0.57307681 0.07798230 0.16092893 0.08037229 0.57719980 0.67828508
## [19] 1.28323340 0.55944450 0.48375054 0.49015205 0.28180908 0.48370863
## [25] 0.50295497 0.63246205 0.05332516 2.99121278 0.25517080 0.48106424
## [31] 1.86192983 0.97139228 0.32350213 1.73428629 0.18641929 0.49786535
## [37] 1.07466123 0.75447785 1.27949060 0.61015489 0.63224445 0.18567429
## [43] 1.47938978 1.21426057 0.59221149 0.22751377 1.79869055 1.06275264
## [49] 0.24253672 0.43661862
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
##  [1] 0.56694311 7.03997608 0.92247764 0.40541145 0.48405035 0.72760225
##  [7] 0.76177102 0.41780714 0.67497980 0.09514301 3.08475805 0.21215004
## [13] 0.17315567 0.10919573 0.07507338 0.14631569 0.48912914 0.66064036
## [19] 0.97488467 0.11729166 0.55600728 0.45056768 0.12338559 0.32568921
## [25] 0.79730807 0.60552564 0.02079892 1.02827914 0.25057467 0.31826715
## [31] 2.31504674 0.63971981 0.12366562 0.94366437 0.07428690 0.14455968
## [37] 1.83035030 0.53391786 0.48415632 0.57309350 0.26818435 0.25413660
## [43] 0.21664096 0.82807245 0.42764694 0.04986452 1.91326423 0.54897038
## [49] 0.09345031 0.23486682
## 
## $squared_prediction_errors
## # A tibble: 50 x 8
##    Population    Income Illiteracy `Life Exp` Murder `HS Grad`   Frost
##         <dbl>     <dbl>      <dbl>      <dbl>  <dbl>     <dbl>   <dbl>
##  1    1.05     0.107       0.243    0.222     0.744     0.127  0.429  
##  2   16.8     26.3         0.0765   8.94      6.31     10.4    1.54   
##  3    0.0656   0.000397    0.118    0.755     0.622     0.644  1.20   
##  4    0.439    0.654       0.00439  0.972     0.178     0.517  0.00805
##  5    9.81     0.626       0.0534   0.00119   0.0275    0.709  1.05   
##  6    0.141    0.0162      0.0108   0.714     0.639     0.237  1.00   
##  7    0.0585   2.86        0.972    0.130     0.0421    0.0436 1.01   
##  8    0.697    0.639       0.0980   0.807     0.159     0.0182 0.460  
##  9    0.00689  0.310       0.357    0.0000264 0.0487    0.0476 1.76   
## 10    0.198    0.0499      0.0142   0.000159  0.144     0.154  0.0153 
## # ... with 40 more rows, and 1 more variable: Area <dbl>
## 
## $adjusted_feature_weights
## Population     Income Illiteracy   Life Exp     Murder    HS Grad 
##  0.0000000  0.0000000  0.4365459  0.2310834  0.3514695  0.2209376 
##      Frost       Area 
##  0.2796782  0.0000000
```

Here we use Ordinary Least Squares regression as the base regressor, 5-fold cross validation for scoring, and a list containg various artifacts from the algorithm:

