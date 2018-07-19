check_class_structure <- function(data, structure, numeric_only = NULL) {

    col_types <- sapply(data, class)
    classes <- structure %in% class(data) %>% any() %>% isTRUE()

    if (isTRUE(numeric_only)) {
        numerics_only <-
            length(col_types[col_types == "numeric"]) == length(col_types)
    } else {
        numerics_only <- NULL
    }

    checks <- c(classes, numerics_only)

    if (FALSE %in% checks) {
        FALSE
    } else {
        TRUE
    }
}

clear_colname_spaces <- function(data, substitute = "") {
    colnames(data) <- gsub(" ", substitute, colnames(data))
    data
}


square_errors <- function(predictions, actual) {

    # inputs must be vectors
    # if (!is.vector(predictions) && !is.vector(actual)) {
    #     stop("arguments need to be vectors")
    # }

    if ("character" %in% sapply(list(predictions, actual), class))
        stop("arguments should be of class numeric or factor")

    # if
    if (class(predictions) == "factor" | class(actual) == "factor") {
        squared_errors <- (as.numeric(predictions) - as.numeric(actual))^2
    } else {
        squared_errors <- (predictions - actual)^2
    }

    return(squared_errors)
}
