#' Gauss-Kronrod nodes/weights on [0,1] for the cumulative-hazard time
#' integral. order=10 uses the exact original hardcoded table (preserves
#' full backward compatibility / exact reproducibility of all earlier
#' results). Any other order uses a genuine n-point Gauss-Legendre rule
#' (Golub-Welsch algorithm) instead - equally accurate for a fixed node
#' count and works for arbitrary n, without adding a numerical-quadrature
#' package dependency (e.g. statmod, pracma) for something this
#' well-defined. Not literally "Kronrod" nodes for orders other than 10 -
#' the name is kept for interface stability, but see gauss_legendre_01().
#'
#' Added to test whether the original 10-node rule's accuracy was a
#' contributing factor to alpha bias observed on specific "hard" datasets
#' during development (see n200 replication investigations) - a fixed
#' 10-node rule may be characteristically less accurate than what other
#' packages use internally for some hazard shapes.
#'
#' @param order Number of quadrature nodes.
gauss_kronrod_nodes <- function(order = 10) {
  if (order == 10) {
    return(list(
      nodes = c(0.013047, 0.067468, 0.160295, 0.283302, 0.425562,
                0.574438, 0.716698, 0.839705, 0.932532, 0.986953),
      weights = c(0.033336, 0.074726, 0.109543, 0.134633, 0.147762,
                  0.147762, 0.134633, 0.109543, 0.074726, 0.033336)
    ))
  }
  gauss_legendre_01(order)
}

#' n-point Gauss-Legendre quadrature nodes/weights on [0,1], via the
#' Golub-Welsch algorithm (eigendecomposition of the tridiagonal Jacobi
#' matrix) - standard, numerically stable, works for arbitrary n.
#'
#' @param n Number of quadrature points; must be at least 2.
gauss_legendre_01 <- function(n) {
  if (n < 2) stop("n must be >= 2")
  k <- 1:(n - 1)
  beta <- k / sqrt(4 * k^2 - 1)
  J <- matrix(0, n, n)
  for (i in seq_along(beta)) {
    J[i, i + 1] <- beta[i]
    J[i + 1, i] <- beta[i]
  }
  eig <- eigen(J, symmetric = TRUE)
  ord <- order(eig$values)
  nodes <- eig$values[ord]
  weights <- 2 * (eig$vectors[1, ord])^2
  list(nodes = (nodes + 1) / 2, weights = weights / 2)
}
