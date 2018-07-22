#' Attribute-wise Learning for Scoring Outliers (ALSO).
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
#' @details  The model_function argument takes a function as its input. For example,
#' either "lm" or lm will work to score using least squares regression. Third-party
#' packages containing the modeling function need to be installed and loaded via
#' library() or referenced (e.g. rpart::rpart).
#'
#' Feature weights are critical in determining outlier scores. Features with high
#' RMSE are given less weight than features with lower RMSE. The adjusted feature
#' weights are found by subtracting the feature RMSE from 1.
#' Feature RMSE values greater than 1 are adjusted to 1, leading to an adjusted
#' feature weight of 0 (no impact on scoring).
#' @references see "Outlier Analysis" (C.C Aggarwal. Springer, 2017) section 7.7
#' @examples
#' # OLS with cross validation for out of sample scoring
#' dtree_also <- ALSO(data = scale(state.x77), model_function = rpart::rpart,
#' cross_validate = TRUE, n_folds = 10, score_only = FALSE)
#'
#' dtree_also$scores
#' dtree_also$squared_prediction_errors
#' dtree_also$feature_rmse
#' dtree_also$feature_weights
#' @importFrom magrittr %<>%
#' @export

ALSO <- function(data, model_function, cross_validate = TRUE,  n_folds = 5,
                 scores_only = TRUE, ...) {

    #
    #
    # Ensure input is a data.frame or tibble data structure comprised of
    # factors, integers, or numeric values, or a numeric matrix. Store a
    # vector of original column names to re-substitute at end
    #
    #


    if (!is.data.frame(data)) {
        stop("Input data must be a data frame or a tibble. Try data = as_tibble(...)")
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
            cv_models <- purrr::map(init_formulas, model_function,
                                    data = training_folds)
            pred <- purrr::map(cv_models, predict, newdata = testing_folds) %>%
                setNames(nm = colnames(data))
        }) %>%
            purrr::map(., bind_cols) %>%  # restore test folds with predictions
            dplyr::bind_rows() %>% #
            dplyr::mutate(fold_id = purrr::flatten_int(folds)) %>% # index rows to match original data
            dplyr::arrange(fold_id) %>%
            dplyr::select(-fold_id)

    } else {

        models <- purrr::map(init_formulas, model_function, data = data)

        predictions <- purrr::map(models, predict) %>%
            setNames(nm = colnames(data))
    }

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
        ifelse(. > 1, 1, .) %>%
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
