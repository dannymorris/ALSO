---
title: "Attribute-Wise Learning for Scoring OUtliers"
author: "Danny Morris"
date: "July 22, 2018"
output:
    html_document:
        keep_md: true                           
---


*Attribute-wise Learning for Scoring Outliers* (ALSO) is an unsupervised anomaly detection algorithm for multidimensional data. The main goal is to automate separate predictive models for each feature in the dataset. In each model, one feature in the dataset is the target and the remaining are predictors. A classifier or regressor is called to model the relationships. Observations are scored and compared to true values, and a numerical outlier score is returned for each observation based on the magnitude of deviation from expected values. Given the vector of numeric outlier scores, simple univariate techniques can be used to extract insights. Larger outlier scores suggest greater outlierness.

# Install


```r
devtools::install_github("dannymorris/ALSO")
```

In *Outlier Analysis* (C.C Aggarwal. Springer, 2017), the author recommends the use of **random forests** as the base regressor/classifier. Random forest are highly robust regressors and classifiers due to bagging and ensembling.

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

## Quick data prep

Currently only data frame or tibbles are supported as inputs to the function.


```r
# z-standardize all columns
zstd <- function(x) {
    (x - mean(x)) / sd(x)
}
```


```r
state_tbl <- state.x77 %>%
    dplyr::as_tibble() %>%
    dplyr::mutate_all(zstd)
```

```
## Warning: package 'bindrcpp' was built under R version 3.4.4
```

## Using the ALSO_RF() function


```r
rf_also <- ALSO::ALSO_RF(data = state_tbl,
                      cross_validate = TRUE,
                      n_folds = 5,
                      scores_only = TRUE)

rf_also
```

```
##  [1] 0.57789957 3.68987961 1.66373073 0.99550531 2.28640189 0.93227561
##  [7] 0.51057386 0.70601997 0.63681528 0.75427265 4.51241328 0.09487573
## [13] 0.63392624 0.10223337 0.18711890 0.13634151 0.28292779 0.65351674
## [19] 1.48923647 0.50837075 0.40109052 0.38397466 0.31937024 0.81841444
## [25] 0.48506953 0.61503422 0.02472821 3.31510751 0.19983167 0.45082050
## [31] 1.90950408 0.72138495 0.31879235 1.45691742 0.12971505 0.43827206
## [37] 0.68491458 0.54271299 1.05786419 0.91161394 0.58848011 0.08348255
## [43] 1.19619643 1.00601584 0.42306435 0.11507177 1.32076308 1.13153669
## [49] 0.20003282 0.78472876
```

Here we return only the outlier scores. Observations are given out-of-sample scores via 5-fold cross validation.


```r
state_tbl %>%
    mutate(also_score = rf_also) %>%
    mutate(state = rownames(state.x77)) %>%
    arrange(desc(also_score)) %>%
    select(state, also_score) 
```

```
## # A tibble: 50 x 2
##    state        also_score
##    <chr>             <dbl>
##  1 Hawaii             4.51
##  2 Alaska             3.69
##  3 Nevada             3.32
##  4 California         2.29
##  5 New Mexico         1.91
##  6 Arizona            1.66
##  7 Maine              1.49
##  8 North Dakota       1.46
##  9 Washington         1.32
## 10 Texas              1.20
## # ... with 40 more rows
```
