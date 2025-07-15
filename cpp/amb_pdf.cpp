#include "amb_pdf.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

static double beta_density(double a, double b, double x) {
    if (x <= 0.0 || x >= 1.0) return 0.0;
    double logB = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    return std::exp((a - 1) * std::log(x) + (b - 1) * std::log(1 - x) - logB);
}

static void xi_and_dxi_da(double a, double u, AmbiguityType type,
                          double& xi, double& dxi_da) {
    double denom = 1.0 - u;
    if (type == AmbiguityType::Standard) {
        double arg  = 2 * (1 - a) / denom - 1;
        double root = std::sqrt(arg);
        xi = 0.5 * (1 - root);
        dxi_da = 1.0 / (2 * denom * root);
    } else {
        double arg  = (1 - a) / denom;
        double root = std::sqrt(arg);
        xi = 0.5 * (1 - root);
        dxi_da = 1.0 / (4 * denom * root);
    }
}

static double h_func(double a, double u, AmbiguityType type, double alpha, double beta, double gamma) {

    double u0 = (type == AmbiguityType::Standard ? std::max(0.0, 2 * a - 1) : 0.0);
    if (u <= u0 || u >= a) return 0.0;

    double f1;
    f1 = beta_density(gamma, alpha + beta, u);

    double xi, dxi_da;
    xi_and_dxi_da(a, u, type, xi, dxi_da);

    double fb = beta_density(alpha, beta, xi)
              + beta_density(alpha, beta, 1 - xi);
    return f1 * dxi_da * fb;
}

static double adaptive_simpson_rec(const std::function<double(double)>& f,
                                   double x0, double x1, double eps,
                                   double S, double f0, double f1, double fm,
                                   int depth) {
    double xm = 0.5 * (x0 + x1);
    double xl = 0.5 * (x0 + xm);
    double xr = 0.5 * (xm + x1);

    double fl = f(xl);
    double fr = f(xr);

    double Sl = (xm - x0) / 6.0 * (f0 + 4 * fl + fm);
    double Sr = (x1 - xm) / 6.0 * (fm + 4 * fr + f1);
    double err = Sl + Sr - S;

    if (std::abs(err) < 15 * eps || depth > 20)
        return Sl + Sr + err / 15.0;

    return adaptive_simpson_rec(f, x0, xm, eps / 2, Sl, f0, fm, fl, depth + 1)
         + adaptive_simpson_rec(f, xm, x1, eps / 2, Sr, fm, f1, fr, depth + 1);
}

double integrate_amb_pdf(double a, double alpha, double beta, double gamma, AmbiguityType type, double tol) {
    if (a <= 0.0 || a >= 1.0)
        throw std::domain_error("Parameter 'a' must lie in (0,1)");

    double u0 = (type == AmbiguityType::Standard ? std::max(0.0, 2 * a - 1) : 0.0);
    auto f = [&](double u) { return h_func(a, u, type, alpha, beta, gamma); };

    double f0 = f(u0);
    double f1 = f(a);
    double fm = f(0.5 * (u0 + a));
    double S  = (a - u0) / 6.0 * (f0 + 4 * fm + f1);

    return adaptive_simpson_rec(f, u0, a, tol, S, f0, f1, fm, 0);
}

void integrate_amb_pdf_batch(const std::vector<double>& a_vals,
                             std::vector<double>& results,
                             double alpha, double beta, double gamma,
                             AmbiguityType type, double tol) {
    results.resize(a_vals.size());
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (size_t i = 0; i < a_vals.size(); ++i)
        results[i] = integrate_amb_pdf(a_vals[i], alpha, beta, gamma, type, tol);
}
