#include "io/config.hpp"
#include "sim/simulation.cuh"
#include <cstdio>
#include <exception>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <config_file>\n", argv[0]);
        return 1;
    }
    try {
        TopologyData topo;
        SimParams params;
        load_config(argv[1], params, topo);

        Simulation sim(params, std::move(topo));
        sim.print_setup();
        if (params.forces_dump_file[0] != '\0') {
            sim.write_forces(params.forces_dump_file);
            printf("Initial forces written to %s\n", params.forces_dump_file);
            return 0;
        }
        sim.run();
        printf("Simulation complete.\n");
    } catch (const std::exception& e) {
        fprintf(stderr, "fastmd: error: %s\n", e.what());
        return 1;
    }
    return 0;
}
