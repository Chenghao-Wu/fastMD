#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>

// Interpolation table as stored for the GPU kernels: data[k] =
// (x_k, f_k, E_k, 0) on a uniform grid x_k = rmin + k * dr, where
//   pair / bond tables:  x = r,          f = F(r) / r   (F = -dE/dr)
//   angle tables:        x = theta [rad], f = -dE/dtheta [energy / rad]
struct TableParams {
    float rmin;
    float rmax;
    float dr;
    float inv_dr;
    int   npoints;
    int   data_offset;
};

struct TableFileData {
    std::vector<TableParams> params;
    std::vector<float4>      data;
};

enum class TableKind { Pair, Bond, Angle };

// How column 4 (and, for angles, column 2) of a table file are to be read.
enum class TableForceConvention {
    Force,             // LAMMPS pair/bond tables: column 4 is F = -dE/dr (default)
    ForceOverR,        // legacy fastMD pair/bond tables: column 4 is F/r
    DegreesPerDegree,  // LAMMPS angle tables: theta in degrees, -dE/dtheta per degree (default)
    Radians,           // legacy fastMD angle tables: theta in radians, -dE/dtheta per radian
};

// Parse a convention keyword: "force" | "force_over_r" (pair/bond) or
// "degrees" | "radians" (angle). Throws on anything else.
TableForceConvention parse_table_convention(const std::string& word, bool angle);

// Read the section `keyword` of a LAMMPS-style table file and convert it to
// the kernel layout above. Validates the point count, monotonic and uniform
// spacing (or "R rlo rhi"), and warns when column 4 is inconsistent with
// -dE/dx under the chosen convention.
TableFileData parse_table_file(const std::string& filename,
                               const std::string& keyword,
                               TableKind kind,
                               TableForceConvention convention);
