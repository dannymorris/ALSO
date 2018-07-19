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
