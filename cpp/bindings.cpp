#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "amb_pdf.hpp"

namespace py = pybind11;

PYBIND11_MODULE(amb_pdf, m) {
    m.doc() = "Adaptive Simpson integration for Ambiguity density";

    py::enum_<AmbiguityType>(m, "AmbiguityType")
        .value("Standard", AmbiguityType::Standard)
        .value("Modified",  AmbiguityType::Modified)
        .export_values();

    m.def("pdf", &integrate_amb_pdf,
          py::arg("a"),
          py::arg("alpha"),
          py::arg("beta"),
          py::arg("gamma"),
          py::arg("type") = AmbiguityType::Standard,
          py::arg("tol")  = 1e-8,
          "Compute ambiguity density at a");

    m.def("pdf_batch",
          [](const std::vector<double>& a_vals,
             double alpha, double beta, double gamma,
             AmbiguityType type,
             double tol) {
              std::vector<double> results;
              integrate_amb_pdf_batch(a_vals, results, alpha, beta, gamma, type, tol);
              return results;
          },
          py::arg("a_vals"),
          py::arg("alpha"),
          py::arg("beta"),
          py::arg("gamma"),
          py::arg("type") = AmbiguityType::Standard,
          py::arg("tol")  = 1e-8,
          "Compute ambiguity density for array of a values");
}
