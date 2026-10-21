# Contributing to fastMD

fastMD is a GPU-resident molecular-dynamics **sampling engine**. We welcome
reproducible bug reports, numerical-validation cases, documentation fixes,
and performance improvements to specialized force-field sampling paths.

## Before reporting a problem

1. Use a current checkout and record its commit SHA (`git rev-parse HEAD`).
2. Record your GPU, NVIDIA driver, CUDA toolkit, compiler, OS, and CMake version.
   `nvidia-smi` and `nvcc --version` are particularly helpful.
3. Make a **small, self-contained** example: config file, initial LAMMPS data
   file (or the commands used to generate it), command line, and output/log.
   Remove any private data before attaching files.
4. State what you expected and what happened, including whether a failure is
   intermittent, size-dependent, or appeared after changing a sampling option.
5. For numerical concerns, report the reference method, integration time
   step, cutoff/skin, precision, thermostat/barostat, and any tolerance used.
   A difference between two trajectories is **not**, on its own, evidence of
   incorrect sampling.
6. For performance concerns, state whether timings include setup, initial
   equilibration, output, and neighbor-list rebuilds; compare like-for-like
   systems at a matched physical and numerical accuracy.

Please report crashes, silent output corruption, GPU leaks across repeated
Python `Simulation` constructions, and unexpected statistical biases
as reliability issues.

## Testing changes

Build and run host-only tests (they do not require a GPU at runtime):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DFASTMD_BUILD_TESTS=ON -DFASTMD_BUILD_PYTHON=ON
cmake --build build -j
ctest --test-dir build -L cpu --output-on-failure
```

On a CUDA-equipped machine also run:

```sh
ctest --test-dir build -L gpu --output-on-failure
```

The public GitHub-hosted CI builds with CUDA 11.8 and 12.4 and runs CPU-labelled
tests. It **does not execute GPU integration tests**. GPU-kernel and sampler
changes require an additional GPU test run before a stable release.

For changes to the sampling loop, add a regression that checks the final
physical state or statistical result; for error handling, add a test that
exercises the failure mode. For kernel performance work, preserve a reference
or finite-difference force/virial check and report end-to-end sampling
throughput, not just one microbenchmark.

## Contributions that preserve fastMD's focus

Favor a small, testable improvement with a measurable sampling benefit over a
large general-purpose feature. In particular:

- Keep particle integration, neighbor maintenance, and force evaluation on
  the GPU during production sampling.
- Keep specialized force-field implementations independent where appropriate:
  different forces need not use the same neighbor representation.
- Prefer regression tests for repeated `Simulation.run(...)` calls,
  restart/recovery, GPU memory lifecycle, and sampled observables.
- Distinguish **in-memory neighbor-overflow rollback** from restart files on
  disk: not all thermostat/integrator and online-observable state is currently
  serialized to the latter, so exact continuation must not be assumed.
