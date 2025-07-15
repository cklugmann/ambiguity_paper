#ifndef AMB_PDF_HPP
#define AMB_PDF_HPP

#include <vector>

enum class AmbiguityType {
    Standard,
    Modified
};

// Single evaluation of the ambiguity PDF via adaptive Simpson
double integrate_amb_pdf(double a,
                         AmbiguityType type = AmbiguityType::Standard,
                         double tol = 1e-8);

// Batch evaluation: returns results in the provided vector
void integrate_amb_pdf_batch(const std::vector<double>& a_vals,
                             std::vector<double>& results,
                             AmbiguityType type = AmbiguityType::Standard,
                             double tol = 1e-8);

#endif // AMB_PDF_HPP
