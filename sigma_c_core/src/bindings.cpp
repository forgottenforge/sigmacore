#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sigma_c/susceptibility.hpp"
#include "sigma_c/stats.hpp"

namespace py = pybind11;

PYBIND11_MODULE(sigma_c_core, m) {
    m.doc() = "Sigma-C Core Performance Engine";

    // Susceptibility Engine
    py::class_<sigma_c::SusceptibilityResult>(m, "SusceptibilityResult")
        .def_readonly("chi", &sigma_c::SusceptibilityResult::chi)
        .def_readonly("sigma_c", &sigma_c::SusceptibilityResult::sigma_c)
        .def_readonly("kappa", &sigma_c::SusceptibilityResult::kappa)
        .def_readonly("smoothed", &sigma_c::SusceptibilityResult::smoothed)
        .def_readonly("baseline", &sigma_c::SusceptibilityResult::baseline);

    py::class_<sigma_c::SusceptibilityEngine>(m, "SusceptibilityEngine")
        .def_static("compute", &sigma_c::SusceptibilityEngine::compute,
            py::arg("epsilon"),
            py::arg("observable"),
            py::arg("kernel_sigma") = 0.6,
            py::arg("edge_damp") = 0.5,
            py::call_guard<py::gil_scoped_release>());

    // Stats Engine
    py::class_<sigma_c::StatsEngine>(m, "StatsEngine")
        .def_static("bootstrap_ci", &sigma_c::StatsEngine::bootstrap_ci,
            py::arg("data"),
            py::arg("n_reps") = 1000,
            py::arg("ci_level") = 0.95,
            py::arg("seed") = 42,
            py::call_guard<py::gil_scoped_release>());
}
