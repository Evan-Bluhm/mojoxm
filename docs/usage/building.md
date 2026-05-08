---
icon: lucide/hammer
---

# Building

mojoxm is a Mojo project. Every example, benchmark, and unit test
compiles to a standalone binary in the repository root. The build is
driven by a single `Makefile`; there is **no CMake, no Bazel, no Python
build glue** in the loop.

## What the build produces

```
./advection_gaussian              # examples/ binaries
./euler_vortex                    # ...
./two_fluid_langmuir              # ...
./bench_advection_3d_p5           # benchmarks/ binaries
./bench_euler_sod_3d_p5           # ...
./euler_3d_test                   # test/ binaries
./limiter_3d_test_p5              # ...
build/mpi_shim.o                  # the only intermediate artefact
```

All binaries land in the **repository root** by convention — the
`.gitignore` excludes them by name pattern. There is no `bin/` directory.

## Building one binary

```bash
make euler_vortex                 # build a single example driver
make bench_euler_sod_3d_p5        # build a single benchmark
make euler_3d_test                # build a single unit test
```

The Makefile rules invoke `mojo build path/to/source.mojo -o name` with
the right include flags. Single-file Mojo build, every time — no
explicit linking, no separate `.o` files (except the C MPI shim).

## Building groups

```bash
make all                          # everything: drivers + benches + tests
make gpu                          # GPU drivers under examples/ only
make cpu                          # the few CPU-only utilities (mpi_hello, etc.)
make test                         # unit tests only
```

## Why no `-j`

Mojo is already multi-threaded **per file**. Empirically:

| Command       | Wall clock for 4-bench batch |
| ------------- | ---------------------------- |
| `make`        | ~12 s                        |
| `make -j4`    | ~18 s                        |

The contention comes from cores already saturated by Mojo's per-file
parallelism. **Just `make`.**

## The MPI shim

The only non-Mojo source file in the build is `src/mpi_shim.c` — a small
C wrapper exposing OpenMPI's variadic `MPI_Init` etc. to Mojo. It builds
to `build/mpi_shim.o`:

```bash
make shim                         # build the shim alone
```

The shim is linked into every MPI-enabled binary (`mpi_hello`, the
test/bench MPI suite, and the 3D `mpi_*` example drivers).

## Troubleshooting

??? failure "`module not found: src.solver`"

    A relative-import path is missing. The Makefile passes
    `-I src` to every Mojo invocation; if you're invoking `mojo build`
    by hand, add it.

??? failure "Build hangs near 100% of one core"

    Mojo's compiler can spin a long time on a generic kernel
    specialisation. Wait a minute or two on a cold build. If it's been
    >5 min on a single file, kill it and report — that's a regression.

??? failure "Stale `build/` after pulling a new branch"

    `make clean` and re-build. Mojo doesn't currently track header
    timestamps the way C compilers do.

??? failure "Linker complains about `pthread_create`"

    The async VTU writer uses pthread directly. The Makefile already
    passes `-lpthread`; if you've copy-pasted a build line, add it.
