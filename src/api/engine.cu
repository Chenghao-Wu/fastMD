#include "engine.hpp"
#include "../io/config.hpp"
#include "../sim/simulation.cuh"
#include <sstream>
#include <stdexcept>

namespace fastmd {

namespace {

std::vector<float> xyz_of(const std::vector<float4>& v) {
    std::vector<float> out(3 * v.size());
    for (size_t i = 0; i < v.size(); i++) {
        out[3 * i] = v[i].x;
        out[3 * i + 1] = v[i].y;
        out[3 * i + 2] = v[i].z;
    }
    return out;
}

std::vector<float3> float3_of(const std::vector<float>& flat, int natoms, const char* what) {
    if (flat.size() != 3 * (size_t)natoms)
        throw std::invalid_argument(std::string(what) + ": expected " + std::to_string(3 * (size_t)natoms) +
                                    " values (natoms x 3), got " + std::to_string(flat.size()));
    std::vector<float3> out(natoms);
    for (int i = 0; i < natoms; i++)
        out[i] = make_float3(flat[3 * i], flat[3 * i + 1], flat[3 * i + 2]);
    return out;
}

}  // namespace

Engine::Engine(std::unique_ptr<Simulation> sim) : sim_(std::move(sim)) {}
Engine::~Engine() = default;

std::unique_ptr<Engine> Engine::from_file(const std::string& config_path) {
    SimParams params;
    TopologyData topo;
    load_config(config_path, params, topo);
    return std::unique_ptr<Engine>(new Engine(std::make_unique<Simulation>(params, std::move(topo))));
}

std::unique_ptr<Engine> Engine::from_text(const std::string& config_text) {
    SimParams params;
    TopologyData topo;
    std::istringstream in(config_text);
    load_config(in, "<config>", params, topo);
    return std::unique_ptr<Engine>(new Engine(std::make_unique<Simulation>(params, std::move(topo))));
}

int Engine::natoms() const { return sim_->params().natoms; }
int Engine::ntypes() const { return sim_->params().ntypes; }
int Engine::step() const { return sim_->current_step(); }
double Engine::dt() const { return sim_->params().dt; }
std::array<double, 6> Engine::box() const {
    const Box& b = sim_->box();
    return {b.L.x, b.L.y, b.L.z, b.xy, b.xz, b.yz};
}

void Engine::run(int nsteps) { sim_->run(nsteps); }
void Engine::run_config() { sim_->run(); }

std::vector<float> Engine::positions() const { return xyz_of(sim_->positions()); }
std::vector<int> Engine::images() const { return sim_->images(); }

std::vector<double> Engine::unwrapped_positions() const {
    std::vector<float4> p = sim_->positions();
    std::vector<int> im = sim_->images();
    const Box& b = sim_->box();
    std::vector<double> out(3 * p.size());
    for (size_t i = 0; i < p.size(); i++) {
        out[3 * i] = (double)p[i].x + (double)im[3 * i] * b.L.x + (double)im[3 * i + 1] * b.xy;
        out[3 * i + 1] = (double)p[i].y + (double)im[3 * i + 1] * b.L.y;
        out[3 * i + 2] = (double)p[i].z + (double)im[3 * i + 2] * b.L.z;
    }
    return out;
}
std::vector<float> Engine::velocities() { return xyz_of(sim_->velocities()); }
std::vector<float> Engine::forces() const { return xyz_of(sim_->forces()); }
std::vector<float> Engine::charges() const { return sim_->charges(); }

std::vector<float> Engine::per_atom_energy() const {
    std::vector<float4> f = sim_->forces();
    std::vector<float> e(f.size());
    for (size_t i = 0; i < f.size(); i++) e[i] = f[i].w;
    return e;
}

std::vector<int> Engine::types() const {
    std::vector<float4> p = sim_->positions();
    std::vector<int> t(p.size());
    for (size_t i = 0; i < p.size(); i++) t[i] = unpack_type_id(p[i].w);
    return t;
}

void Engine::set_positions(const std::vector<float>& xyz, const std::vector<int>* images) {
    sim_->set_positions(float3_of(xyz, natoms(), "set_positions"), images);
}

void Engine::set_velocities(const std::vector<float>& v) {
    sim_->set_velocities(float3_of(v, natoms(), "set_velocities"));
}

void Engine::apply_step_shear(double gamma) { sim_->apply_step_shear((float)gamma); }
void Engine::set_tilt(double xy, bool affine) { sim_->set_tilt((float)xy, affine); }
int Engine::nwalls() const { return sim_->walls().n; }
double Engine::film_center() { return sim_->film_center(); }
void Engine::remove_walls() { sim_->set_walls(WallSet{}); }

Thermo Engine::thermo() {
    ThermoOutput t = sim_->thermo();   // energies per atom
    Thermo out;
    out.kinetic_energy = (double)t.kinetic_energy * natoms();
    out.potential_energy = (double)t.potential_energy * natoms();
    out.temperature = t.temperature;
    for (int k = 0; k < 6; k++) out.pressure[k] = t.stress[k];
    return out;
}

}  // namespace fastmd
