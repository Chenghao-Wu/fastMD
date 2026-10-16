#include "trajectory.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

void TrajectorySchedule::set_linear(int n) {
    if (n < 1) throw std::invalid_argument("trajectory: frame interval must be >= 1");
    mode = Mode::Linear;
    every = n;
}

void TrajectorySchedule::set_log(int block_steps, double b) {
    if (block_steps < 1) throw std::invalid_argument("trajectory: block must be >= 1");
    if (!(b > 1.0)) throw std::invalid_argument("trajectory: log base must be > 1");
    mode = Mode::Log;
    block = block_steps;
    base = b;
    offsets = {0};
    for (double x = 1.0; x < block; x *= base) {
        int o = (int)std::llround(x);
        if (o < block && o != offsets.back()) offsets.push_back(o);
    }
    std::sort(offsets.begin(), offsets.end());
    offsets.erase(std::unique(offsets.begin(), offsets.end()), offsets.end());
}

bool TrajectorySchedule::due(long long step) const {
    switch (mode) {
        case Mode::Linear: return step % every == 0;
        case Mode::Log: return std::binary_search(offsets.begin(), offsets.end(), (int)(step % block));
        default: return false;
    }
}

void TrajectoryWriter::open(const std::string& path, const std::vector<int>& types,
                            const std::vector<int>& mol_ids) {
    close();
    fp_ = fopen(path.c_str(), "wb");
    if (!fp_) throw std::runtime_error("cannot open trajectory file " + path);
    natoms_ = (int)types.size();
    std::vector<int> mol(mol_ids);
    mol.resize(natoms_, 0);
    const char magic[8] = {'F', 'M', 'D', 'T', 'R', 'J', '0', '2'};
    const int32_t head[4] = {2, natoms_, 1, 0};
    fwrite(magic, 1, 8, fp_);
    fwrite(head, sizeof(int32_t), 4, fp_);
    fwrite(types.data(), sizeof(int32_t), natoms_, fp_);
    fwrite(mol.data(), sizeof(int32_t), natoms_, fp_);
    frames_ = 0;
}

void TrajectoryWriter::write_frame_raw(long long step, const double box[6],
                                       const std::vector<float>& xyz, const std::vector<int>& images) {
    if (!fp_) return;
    if ((int)xyz.size() != 3 * natoms_ || (int)images.size() != 3 * natoms_)
        throw std::invalid_argument("trajectory: frame size does not match natoms");
    const int64_t s = step;
    fwrite(&s, sizeof(int64_t), 1, fp_);
    fwrite(box, sizeof(double), 6, fp_);
    fwrite(xyz.data(), sizeof(float), xyz.size(), fp_);
    fwrite(images.data(), sizeof(int32_t), images.size(), fp_);
    fflush(fp_);
    frames_++;
}

void TrajectoryWriter::close() {
    if (fp_) fclose(fp_);
    fp_ = nullptr;
}
