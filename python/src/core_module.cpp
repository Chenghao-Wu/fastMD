// pybind11 binding of fastmd::Engine: the stateful Python interface
// (fastmd.Simulation). Arrays are returned as new numpy arrays in tag order
// (the atom order of the input data file).
#include "api/engine.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>

namespace py = pybind11;
using fastmd::Engine;

namespace {

template <typename T>
py::array_t<T> as_array(std::vector<T>&& v, py::ssize_t cols) {
    auto* heap = new std::vector<T>(std::move(v));
    py::capsule owner(heap, [](void* p) { delete static_cast<std::vector<T>*>(p); });
    const py::ssize_t rows = (py::ssize_t)heap->size() / cols;
    const py::ssize_t item = sizeof(T);
    std::vector<py::ssize_t> shape{rows}, strides{cols * item};
    if (cols > 1) {
        shape.push_back(cols);
        strides.push_back(item);
    }
    return py::array_t<T>(shape, strides, heap->data(), owner);
}

template <typename T>
std::vector<T> flat_rows(const py::array_t<T, py::array::c_style | py::array::forcecast>& a,
                         py::ssize_t natoms, const char* what) {
    if (a.ndim() != 2 || a.shape(0) != natoms || a.shape(1) != 3)
        throw py::value_error(std::string(what) + ": expected an array of shape (" +
                              std::to_string(natoms) + ", 3)");
    return std::vector<T>(a.data(), a.data() + a.size());
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "fastMD GPU molecular dynamics: stateful simulation interface";

    py::class_<fastmd::Thermo>(m, "Thermo")
        .def_readonly("kinetic_energy", &fastmd::Thermo::kinetic_energy)
        .def_readonly("potential_energy", &fastmd::Thermo::potential_energy)
        .def_readonly("temperature", &fastmd::Thermo::temperature)
        .def_property_readonly("pressure", [](const fastmd::Thermo& t) {
            // full 3x3 symmetric tensor from (xx xy xz yy yz zz)
            const double* p = t.pressure;
            std::vector<double> full = {p[0], p[1], p[2], p[1], p[3], p[4], p[2], p[4], p[5]};
            return as_array(std::move(full), 3);
        }, "Pressure tensor (3 x 3).")
        .def_property_readonly("total_energy", [](const fastmd::Thermo& t) {
            return t.kinetic_energy + t.potential_energy;
        })
        .def("__repr__", [](const fastmd::Thermo& t) {
            return "Thermo(ke=" + std::to_string(t.kinetic_energy) + ", pe=" +
                   std::to_string(t.potential_energy) + ", T=" + std::to_string(t.temperature) + ")";
        });

    py::class_<Engine>(m, "Simulation", R"doc(
A fastMD simulation resident on the GPU.

Construct from a config file (``Simulation(path)``) or config text
(``Simulation.from_text(text)``); paths inside the config are relative to the
working directory. Per-particle arrays are (natoms, 3) numpy arrays in the
atom order of the input data file. Positions are wrapped into the box
(sheared by the xy tilt); ``unwrapped_positions`` gives x + ix Lx + iy xy,
y + iy Ly, z + iz Lz.
)doc")
        .def(py::init([](const std::string& path) { return Engine::from_file(path); }),
             py::arg("config"), "Load a config file and set up the simulation.")
        .def_static("from_text", &Engine::from_text, py::arg("config_text"),
                    "Set up a simulation from config text.")
        .def_property_readonly("natoms", &Engine::natoms)
        .def_property_readonly("ntypes", &Engine::ntypes)
        .def_property_readonly("step", &Engine::step, "Time steps taken so far.")
        .def_property_readonly("dt", &Engine::dt)
        .def_property_readonly("box", [](const Engine& e) {
            auto b = e.box();
            return as_array(std::vector<double>(b.begin(), b.end()), 1);
        }, "Current box (Lx, Ly, Lz, xy, xz, yz); changes under NPT.")
        .def_property_readonly("box_length", [](const Engine& e) {
            auto b = e.box();
            if (b[0] != b[1] || b[1] != b[2] || b[3] != 0 || b[4] != 0 || b[5] != 0)
                throw py::value_error("box_length: the box is not cubic; use box");
            return b[0];
        }, "Edge length of a cubic box (use box for other boxes).")
        .def("run", &Engine::run, py::arg("nsteps"), py::call_guard<py::gil_scoped_release>(),
             "Advance nsteps time steps. Outputs configured in the config file are written "
             "when due. Temperature/pressure ramps are defined over the config's nsteps.")
        .def("run_config", &Engine::run_config, py::call_guard<py::gil_scoped_release>(),
             "Run the config's nsteps like the command-line program (progress, final restart, "
             "stress correlation file).")
        .def_property("positions",
                      [](const Engine& e) { return as_array(e.positions(), 3); },
                      [](Engine& e, const py::array_t<float, py::array::c_style | py::array::forcecast>& a) {
                          std::vector<float> x = flat_rows(a, e.natoms(), "positions");
                          e.set_positions(x);
                      },
                      "Wrapped positions (natoms, 3). Assigning wraps the new positions into "
                      "the box, resets image counts to 0 and recomputes forces; use "
                      "set_positions() to keep image counts.")
        .def("set_positions",
             [](Engine& e, const py::array_t<float, py::array::c_style | py::array::forcecast>& a,
                py::object images) {
                 std::vector<float> x = flat_rows(a, e.natoms(), "positions");
                 if (images.is_none()) {
                     e.set_positions(x);
                 } else {
                     auto im = images.cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
                     std::vector<int> v = flat_rows(im, e.natoms(), "images");
                     e.set_positions(x, &v);
                 }
             },
             py::arg("positions"), py::arg("images") = py::none(),
             "Replace positions (wrapped into the box; wrapping adds to the given image "
             "counts) and recompute the neighbor list and forces.")
        .def_property_readonly("images", [](const Engine& e) { return as_array(e.images(), 3); })
        .def_property_readonly("unwrapped_positions",
                               [](const Engine& e) { return as_array(e.unwrapped_positions(), 3); },
                               "Unwrapped positions (natoms, 3), float64, including the box tilt.")
        .def_property("velocities",
                      [](Engine& e) { return as_array(e.velocities(), 3); },
                      [](Engine& e, const py::array_t<float, py::array::c_style | py::array::forcecast>& a) {
                          e.set_velocities(flat_rows(a, e.natoms(), "velocities"));
                      })
        .def_property_readonly("forces", [](const Engine& e) { return as_array(e.forces(), 3); },
                               "Forces at the current positions (natoms, 3).")
        .def_property_readonly("per_atom_energy",
                               [](const Engine& e) { return as_array(e.per_atom_energy(), 1); })
        .def_property_readonly("types", [](const Engine& e) { return as_array(e.types(), 1); })
        .def_property_readonly("charges", [](const Engine& e) { return as_array(e.charges(), 1); })
        .def("apply_step_shear", &Engine::apply_step_shear, py::arg("gamma"),
             "Step shear strain gamma in the xy plane: particles move affinely (x += gamma y, "
             "velocities unchanged) and the box tilt xy grows by gamma Ly. The neighbor list "
             "and forces are recomputed.")
        .def("set_tilt", &Engine::set_tilt, py::arg("xy"), py::arg("affine") = false,
             "Set the box xy tilt. affine=True moves particles with the box (a strain); "
             "otherwise unwrapped positions are kept (e.g. to restore a saved state).")
        .def_property_readonly("nwalls", &Engine::nwalls, "Number of active walls.")
        .def_property_readonly("film_center", &Engine::film_center,
                               "Film reference plane: mass-weighted mean unwrapped coordinate "
                               "along the film normal (needs `film <axis>`).")
        .def("remove_walls", &Engine::remove_walls,
             "Remove all walls (e.g. release a film after confined preparation).")
        .def("thermo", &Engine::thermo,
             "Kinetic and potential energy, temperature and pressure tensor of the current state.");
}
