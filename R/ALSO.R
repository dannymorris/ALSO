#' Attribute-wise Learning for Scoring Outliers (ALSO) with Random Forest.
#'
#' @param data a data.frame, tibble, or numeric matrix
#' @param model_function a model function, e.g. lm (see details)
#' @param cross_validate logical, if TRUE then use k-fold cross-validation for scoring
#' @param n_folds an integer specifying the number of folds if cross_validate = T.
#' Defaults to 5
#' @param scores_only logical, if TRUE return outlier scores only. If FALSE
#' return a list with additional output (see return)
#' @param ... additional arguments passed to the modeling function
#' @return If scores_only = TRUE, a numeric vector of outlier scores is returned.
#' If FALSE, return a list containing outlier scores, squared prediction errors,
#' adjusted feature RMSE (see details), and adjusted feature weights for scoring
#' (see details)
#' @details ALSO_RF() uses random forests (from the ranger package) to compute a
#' number of regressors/classifiers equal to the number of columns in the input
#' dataset. Random forest models are chosen for this method due to their flexibility
#' (use for classification and regression) and robustness. Outlier scores for each
#' observation are determined by computing the aggregate errors from the individual
#' models.
#'
#' Feature weights are critical in determining outlier scores. Features with high
#' RMSE are given less weight than features with lower RMSE. The adjusted feature
#' weights are found by subtracting the feature RMSE from 1.Feature RMSE values
#' greater than 1 are adjusted to 1, leading to an adjusted feature weight of 0
#' (no impact on scoring).
#' @references see "Outlier Analysis" (C.C Aggarwal. Springer, 2017) section 7.7
#' @examples
#' also <- ALSO_RF(data = iris, scores_only = FALSE)#'
#' also$scores
#' also$squared_prediction_errors
#' also$adjusted_feature_weights
#'
#' ALSO_RF(data = iris, num.trees = 100) # pass arguments from ranger::ranger()
#' @importFrom magrittr %<>%
#' @export
ALSO_RF <- function(data, cross_validate = TRUE, n_folds = 5, scores_only = TRUE,
                    ...) {

    #
    #
    # Default base learner is a random forest from the ranger package.
    #
    #

    # if (missing(model_function)) {
    #     message("model_function unspecified. Random forest being used as base learner")
    #     model_function <- ranger::ranger
    # }

    #
    #
    # Ensure input is a data.frame or tibble data structure comprised of
    # factors, integers, or numeric values, or a numeric matrix. Store a
    # vector of original column names to re-substitute at end
    #
    #


    if (!is.data.frame(data)) {
        stop("Input data must be a data.frame or a tibble")
    }

    original_colnames <- colnames(data)

    data %<>%
        clear_colname_spaces() %>%
        mutate_if(is.character, as.factor)


    #
    #
    # Prepare *k* model formulas for *k* features in dataset
    #
    #


    vars <- colnames(data)
    col_classes <- sapply(data, class)

    formulas <- vector("character", length(vars))

    for (i in seq_along(vars)) {
        formulas[i] <- paste(vars[i], "~", ".")
    }

    init_formulas <- purrr:::map(formulas, as.formula)


    #
    #
    # Prepare for cross validation (if necessary), map formulas to the algorithm,
    # make predictions for all *n* observations across *k* features
    #
    #


    if (cross_validate == TRUE) {

        if (is.null(n_folds)) {
            message("n_folds not supplided. Default is 5")
            folds <- caret::createFolds(1:nrow(data), k = 5)
        } else {
            folds <- caret::createFolds(1:nrow(data), k = n_folds)
        }

        predictions <- purrr::map(folds, function(x) {
            training_folds <- data[-x, ]
            testing_folds <- data[x, ]
            cv_models <- purrr::map(init_formulas, ranger::ranger,
                                    data = training_folds, ...)
            cv_model_list <- purrr::map(cv_models, predict, data = testing_folds) %>%
                setNames(nm = colnames(data))

            predictions <- purrr::map(cv_model_list, function(x) x$predictions)
        }) %>%
            purrr::map(., bind_cols) %>%  # restore test folds with predictions
            dplyr::bind_rows() %>% #
            dplyr::mutate(fold_id = purrr::flatten_int(folds)) %>% # index rows to match original data
            dplyr::arrange(fold_id) %>%
            dplyr::select(-fold_id)

    } else {

        models <- purrr::map(init_formulas, ranger::ranger, data = data, ...)

        model_list <- purrr::map(models, predict, data = data) %>%
            setNames(nm = colnames(data))

        predictions <- purrr::map(model_list, function(x) x$predictions)
    }

    # if user-specified model function

    # if (user_model) {
    #     myf <- function(formula, data) {
    #         m <- rpart::rpart(formula = formula, data = data)
    #         p <- predict(m)
    #
    #         list(data = data,
    #              predictions = p)
    #     }
    #
    #     predictions <- purrr::map(init_formulas, function(x) {
    #         myf(x, data = data)$predictions
    #     }) %>%
    #         setNames(nm = colnames(data))
    # }

    #
    #
    # Compute squared prediction errors, summarize each feature by its root mean
    # squared error, derive feature weights based on rmse, and  calculate outlier
    # scores
    #
    #

    squared_prediction_errors <- purrr::map2(predictions, as.list(data), square_errors) %>%
        dplyr::bind_cols()

    adjusted_feature_rmse <- purrr::map_dbl(squared_prediction_errors,
                                            function(x) mean(x) %>% sqrt) %>%
        ifelse(. > 1, 1, .)

    adjusted_feature_weights <- adjusted_feature_rmse %>%
        purrr::map_dbl(., function(x) 1 - x)

    outlier_scores <- purrr::map2(squared_prediction_errors, adjusted_feature_weights,
                                  function(x, y) x * y) %>%
        purrr::reduce(`+`)

    #
    #
    # Output
    #
    #

    if (scores_only == TRUE) {
        return(outlier_scores)
    } else {
        return(
            list(
                scores = outlier_scores,
                squared_prediction_errors = squared_prediction_errors,
                adjusted_feature_weights = adjusted_feature_weights
            ) %>%
                purrr::map_at(., 2:4, function(x) setNames(x, nm = original_colnames))
        )
    }
}
