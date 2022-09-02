#!/usr/bin/env Rscript
rm(list = ls())
n_pkgs <- c("vars", "bruceR", "rjson", "VLTimeCausality", "lmtest", "NlinTS")
invisible(capture.output(args <- commandArgs(trailingOnly = TRUE)))

invisible(capture.output(lapply(n_pkgs, FUN = function(x) {
  #require(x, character.only = TRUE, quietly = TRUE, attach.required = FALSE)
  if (!x %in% utils::installed.packages()) {
    invisible(capture.output(install.packages(x, dependencies = TRUE, repos = "https://cran.us.r-project.org")))
  }
})))

times_series <- read.csv(args[1])

paramters <- rjson::fromJSON(file = args[2])

# make the paramters from the .json in R available
for (o in names(paramters)) {
  invisible(capture.output(assign(o, paramters[[o]])))
}

set.seed(seed)

########################################################################################################################
### Section: Tests
########################################################################################################################

###
### Subsection: Bi-variate GC
###

bi_variate_gc <- function(tds) {
  for (l in seq_len(p)) {
    # vars
    for (t in seq_along(ts_names)) {
      for (c in seq_len(k)[-t]) {
        invisible(capture.output(tmp_var <- vars::VAR(times_series[, c(t, c)], p = l, type = "const")))
        r <- if (c > t) c - 1 else c
        invisible(capture.output(tds$vars[[l]][r, t] <- vars::causality(tmp_var, cause = paste0("t", c))$
          Granger$
          p.value))
      }
    }
    # lmtest
    # bi_variate Granger-causality tests
    for (t in seq_along(ts_names)) {
      for (c in seq_len(k)[-t]) {
        r <- if (c > t) c - 1 else c
        invisible(capture.output(tds$lmtest[[l]][r, t] <- lmtest::grangertest(times_series[, c],
                                                                              times_series[, t],
                                                                              order = l)[["Pr(>F)"]][2]))
      }
    }
    #VLTimeCausality
    for (t in seq_along(ts_names)) {
      for (c in seq_len(k)[-t]) {
        r <- if (c > t) c - 1 else c
        invisible(capture.output(tds$VLTimeCausality[[l]][r, t] <- VLTimeCausality::VLGrangerFunc(
          # tests wether X causes Y
          Y = times_series[, t],
          X = times_series[, c],
          alpha = 0.05,
          maxLag = l,
          gamma = 0,
          autoLagflag = FALSE
        )$p.val))
      }
    }
    #
    for (t in seq_along(ts_names)) {
      for (c in seq_len(k)[-t]) {
        r <- if (c > t) c - 1 else c
        # The null hypothesis of this test is that the second time series does not cause the first one.
        invisible(capture.output(tds$NlinTS[[l]][r, t] <- NlinTS::nlin_causality.test(ts1 = times_series[, t],
                                                                                      ts2 = times_series[, c], lag = l,
                                                                                      LayersUniv = 1, LayersBiv = 1,
                                                                                      seed = seed)$pvalue))
      }
    }
  }
  return(tds)

}

###
### Subsection: Multivariate GC
###

multivariate_gc <- function(tds) {
  for (l in seq_len(p)) {
    # lmtest
    for (t in seq_along(ts_combinations)) {
      for (c in seq_along(ts_combinations[[t]])) {
        elements <- ts_combinations[[t]][[c]]
        selection <- ts_names %in% c(ts_names[t], elements)
        invisible(capture.output(tmp_var <- vars::VAR(times_series[, selection], p = l, type = "const")))
        invisible(capture.output(tds$vars[[l]][c, t] <- vars::causality(tmp_var, cause = elements)$
          Granger$
          p.value))
      }
    }
    # bruceR
    invisible(capture.output(bruceR_var <- vars::VAR(times_series, p = l)))
    invisible(capture.output(bruceR_var_res <- summary(bruceR_var)))
    bruceR_coef_mat <- bruceR_p_value_mat <- NULL
    for (ts in bruceR_var_res$varresult) {
      tmp_coef <- ts$coefficients
      rm_const <- -nrow(tmp_coef)
      bruceR_coef_mat <- cbind(bruceR_coef_mat, tmp_coef[rm_const, 1])
      bruceR_p_value_mat <- cbind(bruceR_p_value_mat, tmp_coef[rm_const, 4])
    }

    invisible(capture.output(bruceR_gc <- bruceR::granger_causality(bruceR_var)))
    tds$bruceR[[l]] <- matrix(bruceR_gc$result$p.F, ncol = k)
  }
  return(tds)
}

###
### Subsection: All-on-one GC
###

all_on_one_gc <- function(tds) {
  for (l in seq_len(p)) {
    # partial multivariate Granger-causality tests
    invisible(capture.output(tmp_var <- vars::VAR(times_series, p = l)))
    for (t in seq_along(ts_names)) {
      # vcov. = vcovHC(vars_var)
      invisible(capture.output(tds$vars[[l]][1, t] <- vars::causality(tmp_var, cause = ts_combinations[[t]])$
        Granger$
        p.value))
    }
  }
  return(tds)
}

########################################################################################################################
### Section: Granger causality
########################################################################################################################

###
### Subsection: Datatype preperation
###

data_cont <- replicate(p, matrix(NA, nrow = c_len, ncol = k), simplify = FALSE)
gc_results <- replicate(length(packages),
                        data_cont,
                        simplify = FALSE)
names(gc_results) <- packages
if (test_type == "bi-variate") {
  gc_results <- bi_variate_gc(gc_results)
}else if (test_type == "multivariate") {
  gc_results <- multivariate_gc(gc_results)
}else {
  gc_results <- all_on_one_gc(gc_results)
}

########################################################################################################################
### Subsection: Export all results (and import them back into Python)
########################################################################################################################
write(rjson::toJSON(gc_results), paste0(cwd, "/", r_filename))
