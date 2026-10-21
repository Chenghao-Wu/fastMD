#include "table_parser.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

TableForceConvention parse_table_convention(const std::string& w, bool angle) {
    if (!angle && w == "force") return TableForceConvention::Force;
    if (!angle && w == "force_over_r") return TableForceConvention::ForceOverR;
    if (angle && w == "degrees") return TableForceConvention::DegreesPerDegree;
    if (angle && w == "radians") return TableForceConvention::Radians;
    throw std::runtime_error("unknown table convention '" + w + "' (expected " +
                             (angle ? "degrees|radians" : "force|force_over_r") + ")");
}

TableFileData parse_table_file(const std::string& filename,
                               const std::string& keyword,
                               TableKind kind,
                               TableForceConvention conv) {
    const bool angle = (kind == TableKind::Angle || kind == TableKind::Dihedral);   // angular abscissa
    if (angle != (conv == TableForceConvention::DegreesPerDegree ||
                  conv == TableForceConvention::Radians))
        throw std::runtime_error("table convention does not match the table kind");
    const std::string where = filename + " [" + keyword + "]";

    std::ifstream in(filename);
    if (!in.is_open()) throw std::runtime_error("Cannot open table file: " + filename);

    // Find the section: a line whose first word is the keyword.
    std::string line;
    bool found = false;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::string first;
        if ((iss >> first) && first[0] != '#' && first == keyword) { found = true; break; }
    }
    if (!found) throw std::runtime_error("Table keyword not found: " + keyword + " in " + filename);

    // Parameter line: N n [R rlo rhi | RSQ ... | BITMAP ... | FP ... | EQ ...]
    int n = 0;
    bool has_r = false;
    double rlo = 0, rhi = 0;
    while (std::getline(in, line)) {
        std::istringstream hss(line.substr(0, line.find('#')));
        std::string word;
        if (!(hss >> word)) continue;
        if (word != "N") throw std::runtime_error(where + ": expected 'N <n> ...' parameter line");
        if (!(hss >> n) || n < 2) throw std::runtime_error(where + ": N must be >= 2");
        while (hss >> word) {
            if (word == "R") {
                if (!(hss >> rlo >> rhi) || !(rhi > rlo))
                    throw std::runtime_error(where + ": 'R rlo rhi' needs rhi > rlo");
                has_r = true;
            } else if (word == "RSQ" || word == "BITMAP") {
                throw std::runtime_error(where + ": '" + word + "' spacing is not supported; "
                                         "provide a table uniform in r (or use 'R rlo rhi')");
            } else if (word == "FP") {
                double a, b;
                hss >> a >> b;   // derivative hints for spline styles: not needed
            } else if (word == "EQ") {
                double a;
                hss >> a;        // equilibrium value (bond/angle minimizer hint)
            } else {
                throw std::runtime_error(where + ": unknown table parameter '" + word + "'");
            }
        }
        break;
    }
    if (n == 0) throw std::runtime_error(where + ": missing 'N' parameter line");

    std::vector<double> x(n), e(n), f(n);
    int got = 0;
    while (got < n && std::getline(in, line)) {
        std::istringstream iss(line.substr(0, line.find('#')));
        long idx;
        double xv, ev, fv;
        if (!(iss >> idx)) continue;  // blank / comment line
        if (!(iss >> xv >> ev >> fv) || !std::isfinite(xv) || !std::isfinite(ev) || !std::isfinite(fv))
            throw std::runtime_error(where + ": bad data line '" + line + "'");
        x[got] = xv; e[got] = ev; f[got] = fv;
        got++;
    }
    if (got != n)
        throw std::runtime_error(where + ": expected " + std::to_string(n) + " data lines, found " +
                                 std::to_string(got));

    if (has_r) {
        // LAMMPS semantics: with R the abscissae are recomputed uniformly.
        double maxdev = 0;
        for (int k = 0; k < n; k++) {
            double xr = rlo + (rhi - rlo) * k / (n - 1);
            maxdev = std::max(maxdev, std::fabs(xr - x[k]) / std::max(std::fabs(xr), 1e-30));
            x[k] = xr;
        }
        if (maxdev > 1e-6)
            fprintf(stderr, "Warning: %s: column-2 values differ from the 'R' grid by up to %.2g "
                            "(relative); using the R grid\n", where.c_str(), maxdev);
    } else {
        for (int k = 1; k < n; k++)
            if (!(x[k] > x[k - 1]))
                throw std::runtime_error(where + ": table abscissae must increase strictly");
        double dx = (x[n - 1] - x[0]) / (n - 1);
        for (int k = 1; k < n; k++)
            if (std::fabs((x[k] - x[k - 1]) - dx) > 1e-3 * dx)
                throw std::runtime_error(where + ": table spacing is not uniform (kernels interpolate "
                                         "on a uniform grid); resample or add 'R rlo rhi'");
    }

    // Convert to kernel units: radians for angles; F/r for pair and bond.
    if (conv == TableForceConvention::DegreesPerDegree) {
        const double deg2rad = M_PI / 180.0;
        for (int k = 0; k < n; k++) {
            x[k] *= deg2rad;
            f[k] /= deg2rad;   // -dE/dtheta per degree -> per radian
        }
    }
    if (kind == TableKind::Dihedral) {
        // one full period [-180, 180] degrees; the kernel interpolates phi = atan2(...) in (-pi, pi]
        if (std::fabs(x[0] + M_PI) > 1e-5 || std::fabs(x[n - 1] - M_PI) > 1e-5)
            throw std::runtime_error(where + ": dihedral table must span exactly [-180, 180] degrees");
        double emax = 0;
        for (int k = 0; k < n; k++) emax = std::max(emax, std::fabs(e[k]));
        if (std::fabs(e[0] - e[n - 1]) > 1e-6 * (1.0 + emax) || std::fabs(f[0] - f[n - 1]) > 1e-6 * (1.0 + emax))
            throw std::runtime_error(where + ": dihedral table is not periodic (rows at -180 and 180 "
                                     "degrees differ in energy or force)");
    } else if (angle) {
        if (x[0] < -1e-6 || x[n - 1] > M_PI + 1e-6)
            throw std::runtime_error(where + ": angle table must lie within [0, 180] degrees");
    } else if (x[0] <= 0.0) {
        throw std::runtime_error(where + ": table distances must be > 0");
    }

    // Consistency of the force column with -dE/dx (central differences),
    // evaluated on the physical force (F, not F/r).
    std::vector<double> force_phys(f);
    if (conv == TableForceConvention::ForceOverR)
        for (int k = 0; k < n; k++) force_phys[k] = f[k] * x[k];
    double fmax = 0, devmax = 0;
    for (int k = 1; k + 1 < n; k++) {
        double num = -(e[k + 1] - e[k - 1]) / (x[k + 1] - x[k - 1]);
        devmax = std::max(devmax, std::fabs(num - force_phys[k]));
        fmax = std::max(fmax, std::fabs(force_phys[k]));
    }
    if (n >= 3 && fmax > 0 && devmax > 0.05 * fmax) {
        const char* hint = angle ? "check degrees vs radians (angle_table ... degrees|radians)"
                                 : "check F vs F/r (table ... force|force_over_r)";
        fprintf(stderr, "Warning: %s: force column deviates from -dE/dx by up to %.3g (%.0f%% of "
                        "max |F|); %s\n", where.c_str(), devmax, 100.0 * devmax / fmax, hint);
    }

    TableFileData result;
    result.data.resize(n);
    for (int k = 0; k < n; k++) {
        double fk = f[k];
        if (conv == TableForceConvention::Force) fk = f[k] / x[k];
        result.data[k] = make_float4((float)x[k], (float)fk, (float)e[k], 0.0f);
    }
    TableParams tp;
    tp.rmin = (float)x[0];
    tp.rmax = (float)x[n - 1];
    tp.npoints = n;
    tp.dr = (float)((x[n - 1] - x[0]) / (n - 1));
    tp.inv_dr = 1.0f / tp.dr;
    tp.data_offset = 0;
    tp.flags = 0;
    result.params.push_back(tp);
    return result;
}
