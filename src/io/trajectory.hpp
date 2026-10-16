#pragma once
// fastMD trajectory format (version 2), little-endian, all in tag order (the
// atom order of the input data file):
//
//   header: char magic[8] = "FMDTRJ02"; int32 version (2); int32 natoms;
//           int32 flags (bit 0: image counts present); int32 reserved;
//           int32 type[natoms]; int32 mol_id[natoms]
//   frame:  int64 step; float64 box[6] = {Lx, Ly, Lz, xy, xz, yz};
//           float32 pos[natoms][3]   (wrapped into the box)
//           int32   image[natoms][3] (if flags & 1)
//
// Unwrapped coordinates: x + ix Lx + iy xy + iz xz, y + iy Ly + iz yz,
// z + iz Lz. Frames have a fixed size, so frame k starts at
// header_bytes + k * frame_bytes (python/fastmd/trajectory.py reads it).
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct TrajectorySchedule {
    enum class Mode { Off, Linear, Log } mode = Mode::Off;
    int every = 0;               // Linear: frame every `every` steps
    int block = 0;               // Log: blocks of `block` steps ...
    double base = 2.0;           // ... with frames at offsets 0 and round(base^j)
    std::vector<int> offsets;    // Log: sorted distinct offsets in [0, block)

    void set_linear(int every);
    void set_log(int block, double base);
    bool due(long long step) const;
};

class TrajectoryWriter {
public:
    ~TrajectoryWriter() { close(); }
    void open(const std::string& path, const std::vector<int>& types, const std::vector<int>& mol_ids);
    // pos: xyz (w ignored) in tag order; images: [3 natoms] in tag order
    template <typename Float4>
    void write_frame(long long step, const double box[6], const std::vector<Float4>& pos,
                     const std::vector<int>& images) {
        std::vector<float> xyz(3 * pos.size());
        for (size_t i = 0; i < pos.size(); i++) {
            xyz[3 * i] = pos[i].x;
            xyz[3 * i + 1] = pos[i].y;
            xyz[3 * i + 2] = pos[i].z;
        }
        write_frame_raw(step, box, xyz, images);
    }
    void close();
    bool is_open() const { return fp_ != nullptr; }
    long long frames() const { return frames_; }

private:
    void write_frame_raw(long long step, const double box[6], const std::vector<float>& xyz,
                         const std::vector<int>& images);
    FILE* fp_ = nullptr;
    int natoms_ = 0;
    long long frames_ = 0;
};
