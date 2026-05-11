#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>

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

TableFileData parse_table_file(const std::string& filename,
                                const std::string& keyword);
