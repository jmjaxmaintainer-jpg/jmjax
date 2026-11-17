#' Random-effects standard deviations and correlations
#'
#' The standard deviation of each random effect, named after the columns of
#' the random-effects design (e.g. \code{(Intercept)}, \code{time},
#' \code{I(time^2)}), and their correlation matrix. For \code{jm_bayes()}
#' fits these are posterior means (each correlation's own posterior mean);
#' for \code{jm_mle()} fits, the maximum-likelihood estimates. Use
#' \code{summary(fit)} for intervals; the entries there are
#' \code{sigma_b0}, \code{sigma_b1}, ... and \code{rho} (two random effects)
#' or \code{rho_i_j} (three or more, 0-based, \code{i < j}).
#'
#' @param x A \code{"jmjax"} fit.
#' @param sigma Ignored (present for compatibility with the
#'   \code{nlme::VarCorr} generic).
#' @param ... Ignored.
#' @return An object of class \code{"VarCorr.jmjax"}: a list with
#'   \code{sd} (named vector), \code{corr} (q x q matrix, or \code{NULL}
#'   for a single random effect) and \code{cov} (q x q matrix). Without a
#'   correlation in the model (\code{random_effects_corr = FALSE}),
#'   \code{corr} is the identity.
#' @examples
#' \dontrun{
#' fit <- jm_bayes(y ~ time + I(time^2), Surv(time, event) ~ 1,
#'                 data_long, data_surv, id_var = "id", time_var = "time",
#'                 random_formula = ~ time + I(time^2))
#' VarCorr(fit)
#' }
#' @export
VarCorr.jmjax <- function(x, sigma = 1, ...) {
  est <- x$estimates
  nm <- names(est)
  # Look entries up by name: [[ on a named numeric vector errors for a
  # missing name instead of returning NULL.
  get <- function(k) if (k %in% nm) unname(est[[k]]) else NULL
  if (!is.null(get("sigma_b")) && !any(grepl("^sigma_b[0-9]+$", nm))) {
    sd <- get("sigma_b")
  } else {
    idx <- sort(as.integer(sub("^sigma_b", "", grep("^sigma_b[0-9]+$", nm, value = TRUE))))
    sd <- vapply(idx, function(k) get(paste0("sigma_b", k)), numeric(1))
  }
  q <- length(sd)
  if (!q) return(NULL)
  re_names <- x$model_info$re_names
  if (length(re_names) != q) re_names <- paste0("b", seq_len(q) - 1L)
  names(sd) <- re_names

  corr <- NULL
  if (q >= 2L) {
    corr <- diag(q)
    if (q == 2L && !is.null(get("rho"))) {
      corr[1, 2] <- corr[2, 1] <- get("rho")
    } else {
      for (i in seq_len(q - 1L) - 1L) {
        for (j in (i + 1L):(q - 1L)) {
          v <- get(sprintf("rho_%d_%d", i, j))
          if (!is.null(v)) corr[i + 1L, j + 1L] <- corr[j + 1L, i + 1L] <- v
        }
      }
    }
    dimnames(corr) <- list(re_names, re_names)
  }
  cov <- if (is.null(corr)) matrix(sd^2, 1, 1) else (sd %o% sd) * corr
  dimnames(cov) <- list(re_names, re_names)
  structure(list(sd = sd, corr = corr, cov = cov,
                 kind = if (!is.null(x$posterior_samples)) "posterior mean" else "estimate"),
            class = "VarCorr.jmjax")
}

#' @export
print.VarCorr.jmjax <- function(x, digits = 3, ...) {
  out <- data.frame(SD = formatC(x$sd, digits = digits, format = "f"),
                    row.names = names(x$sd), check.names = FALSE)
  if (!is.null(x$corr)) {
    q <- length(x$sd)
    cm <- matrix("", q, q - 1L)
    for (i in seq_len(q)[-1]) {
      for (j in seq_len(i - 1L)) cm[i, j] <- formatC(x$corr[i, j], digits = 2, format = "f")
    }
    colnames(cm) <- c("Corr", rep("", q - 2L))
    out <- cbind(out, as.data.frame(cm, stringsAsFactors = FALSE))
    names(out) <- c("SD", "Corr", if (q > 2L) rep(" ", q - 2L))
  }
  print(out, quote = FALSE, right = TRUE)
  invisible(x)
}

#' @importFrom nlme VarCorr
#' @export
nlme::VarCorr
