#include "table_parser.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>

TableFileData parse_table_file(const std::string& filename,
                                const std::string& keyword) {
    std::ifstream in(filename);
    if (!in.is_open())
        throw std::runtime_error("Cannot open table file: " + filename);

    std::string line;
    bool found_keyword = false;
    int n = 0;
    float rlo = 0.0f, rhi = 0.0f;
    bool has_r = false;

    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string first;
        iss >> first;
        if (first == keyword) {
            found_keyword = true;
            while (std::getline(in, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream hss(line);
                std::string key;
                hss >> key;
                if (key == "N") {
                    hss >> n;
                    std::string r_key;
                    if (hss >> r_key) {
                        if (r_key == "R") {
                            hss >> rlo >> rhi;
                            has_r = true;
                        }
                    }
                }
                break;
            }
            break;
        }
    }

    if (!found_keyword)
        throw std::runtime_error("Table keyword not found: " + keyword);

    std::vector<float4> points;
    points.reserve(n);

    while (std::getline(in, line) && (int)points.size() < n) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int idx;
        float r, e, f;
        iss >> idx >> r >> e >> f;
        points.push_back(make_float4(r, f, e, 0.0f));
    }

    if ((int)points.size() != n)
        throw std::runtime_error("Table data line count mismatch in " + filename);

    for (size_t i = 1; i < points.size(); ++i) {
        if (points[i].x <= points[i - 1].x)
            throw std::runtime_error("Table r values not monotonically increasing");
    }

    if (!has_r) {
        rlo = points.front().x;
        rhi = points.back().x;
    }

    TableParams tp;
    tp.rmin = rlo;
    tp.rmax = rhi;
    tp.npoints = n;
    tp.dr = (rhi - rlo) / (n - 1);
    tp.inv_dr = 1.0f / tp.dr;
    tp.data_offset = 0;

    TableFileData result;
    result.params.push_back(tp);
    result.data = std::move(points);
    return result;
}
