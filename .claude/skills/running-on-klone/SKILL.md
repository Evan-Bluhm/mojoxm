---
name: running-on-klone
description: How to build, run, and debug mojoxm on UW Klone / Hyak via `scripts/klone-run`. Use whenever the user mentions Klone, Hyak, the cluster, SLURM, srun, salloc, apptainer, a `.sif` container, multi-GPU runs, nsys profiling on the cluster, or issues like "job stuck in queue", "only one GPU", "NP=N doesn't do anything", CUDA_ERROR_NO_BINARY_FOR_GPU, libxpmem, or UCX errors. Also use when preparing a new driver or profiling wrapper that needs to dispatch through klone-run.
---

<!-- EDITORIAL GUIDELINES
This skill is a correction layer: it encodes site-specific facts (Klone
partitions, GRES types, quotas, login-node restrictions) and
project-specific facts (scripts/klone-run, mojo.sif, driver classification)
that a pretrained model will not know. Be terse. Use tables. Don't
re-explain general SLURM or Apptainer concepts that models already handle.
Only document things that would surprise a reader who knows SLURM generally
but has never seen Klone or mojoxm specifically.
-->

Klone is UW-IT's SLURM + Apptainer cluster. mojoxm builds and runs there via
`scripts/klone-run`, which wraps srun + apptainer + mpirun into one command.
Build happens **inside** the srun allocation so Mojo's generated GPU PTX
matches the actual GPU (otherwise you hit `CUDA_ERROR_NO_BINARY_FOR_GPU` when
SLURM reassigns node types between build and run).

## Session setup

```bash
ssh klone-login               # lands on klone-login01 (persistent)
cd /gscratch/aaplasma/embluhm/code/mojoxm
conda-activate                # function from ~/.bashrc; sets up python env
```

- `/gscratch/aaplasma/…` is scratch; put everything (source, `mojo.sif`,
  simulation output) here. **Do not write output to `$HOME`** — home has
  small quotas that fill fast with VTU frames or nsys traces.
- The `module` command works but emits "not supported on Klone Login nodes"
  and does nothing useful. Ignore it; `klone-run` loads modules by setting
  absolute-path `PATH` / `LD_LIBRARY_PATH` inside the srun heredoc.
- `nvcc`, `mpirun`, and GPU libraries do not exist on the login node. Any
  command that needs them must run inside an srun (klone-run handles this).

## Dispatch: scripts/klone-run

Defaults:

| Var              | Default                      | Notes                                                                  |
|------------------|------------------------------|------------------------------------------------------------------------|
| `NP`             | 1                            | MPI ranks **and** GPU count for `mpi_*` drivers; ignored otherwise     |
| `GPU`            | a40                          | `a40 \| a100 \| l40 \| l40s \| rtx6k \| 2080ti \| p100 \| h200`        |
| `CONSTRAINT`     | empty                        | e.g. `a40\|rtx6k\|l40\|l40s\|a100`; wins over `GPU` when set           |
| `TIME`           | 30:00                        | SLURM walltime                                                         |
| `ACCOUNT`        | aaplasma                     | also valid: `aaplasma-ckpt`                                            |
| `PARTITION`      | ckpt                         | see Partitions table below                                             |
| `MEM`            | 16G                          |                                                                        |
| `SIF`            | `$PROJECT_DIR/mojo.sif`      | build once with `apptainer build --fakeroot mojo.sif mojo.def`         |
| `NO_BUILD`       | 0                            | `1` skips the `make` step (use the pre-built binary)                   |
| `PROFILE_WRAPPER`| empty                        | path to a wrapper spliced between mpirun and apptainer (e.g. nsys)     |

Common recipes:

```bash
# Single-rank GPU driver
scripts/klone-run euler_taylor_green

# Multi-rank MPI driver, one GPU per rank
NP=4 scripts/klone-run mpi_advection_gaussian

# Broaden GPU selection when ckpt's default (a40) is busy
NP=4 CONSTRAINT='a40|rtx6k|l40|l40s|a100' scripts/klone-run mpi_advection_gaussian

# Nsight Systems profiling via a host-side wrapper
NP=4 PROFILE_WRAPPER=/gscratch/aaplasma/embluhm/nsys_wrapper.sh \
    scripts/klone-run mpi_advection_gaussian

# Pinned GPU type, longer walltime
NP=8 GPU=a100 TIME=2:00:00 scripts/klone-run mpi_advection_gaussian

# Skip rebuild (when you've just run and only changed runtime env)
NO_BUILD=1 scripts/klone-run mpi_advection_gaussian
```

The `===> srun …` echo line shows the full srun request — check it to confirm
`--gres=gpu:<type>:<N>` matches your intent. The subsequent
`===> mpirun -np N --oversubscribe [gpu-pin] apptainer exec …` line confirms
the per-rank GPU pinner was spliced in (each rank gets
`CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK`).

## Partitions

| Partition        | GPU types available                                  | Notes                                                |
|------------------|------------------------------------------------------|------------------------------------------------------|
| `ckpt`           | 2080ti, a40, a100, p100, rtx6k                       | **default**; preemptible; fastest queue              |
| `ckpt-all`       | 2080ti, a40, a100, h200, l40, l40s, p100, rtx6k      | like ckpt + newer GPUs; preemptible                  |
| `gpu-a40`        | a40 only                                             | strict queue, dedicated aaplasma nodes; slower queue |
| `gpu-rtx6k` etc. | one type each                                        | strict queues, one per GPU type                      |

`ckpt` does **not** include l40, l40s, or h200 — if you want those, use
`PARTITION=ckpt-all`. `ckpt` / `ckpt-all` happily dispatch to any idle GPU
type; use `CONSTRAINT` to narrow rather than pinning `GPU=` when you don't
care which GPU you get.

## Driver classification (hard-coded in klone-run)

| Driver                  | IS_MPI | NEEDS_GPU | Notes                                          |
|-------------------------|--------|-----------|-------------------------------------------------|
| `mpi_hello`             | 1      | 0         | pure MPI smoke test, login-node-safe if built  |
| `mpi_partition`         | 1      | 0         | pure MPI smoke test                            |
| `mpi_patch_mesh`        | 1      | 1         | GPU-resident PatchMesh build                   |
| `mpi_halo_pingpong`     | 1      | 1         | end-to-end halo exchange                       |
| `mpi_advection_gaussian`| 1      | 1         | full multi-rank DG solve                       |
| everything else         | 0      | 1         | single-rank GPU drivers (NP ignored, warning)  |

Classification is by name prefix: `mpi_*` → IS_MPI=1. A non-MPI driver with
NP>1 prints a warning and still runs a single process (the driver never calls
`mpi.init()` so extra ranks would do nothing anyway).

## Verifying a run

```bash
squeue -u $USER                                    # queue state
scontrol show job <jobid> | grep -E "State|Gres|NodeList|TresPerNode"   # while running
sacct -j <jobid> -o JobID,JobName,Partition,AllocTRES,State             # after run (scontrol forgets)
scontrol show node <nodename> | grep -E "Gres|Partitions"
sinfo -p ckpt -t idle -o "%P %G %N" | head        # idle nodes + their GPUs
ssh <nodename>                                     # you can ssh into nodes that run your jobs
    ps aux | grep $USER                            # see what's actually running there
```

`TresPerNode=gres/gpu:N` in `scontrol show job` is the ground truth for GPU
count. The per-rank GPU assignment happens via env (not SLURM-visible), so
confirm it by checking that wall time scales with NP — a 4× speedup going
from NP=1 to NP=4 on a 4-GPU allocation means pinning worked; a ≤1× "speedup"
means every rank is fighting for the same device.

## Building the Apptainer image (one-time)

Mojo's wheel requires glibc ≥ 2.34; Klone's host glibc is 2.28. `mojo.sif` is
an Ubuntu 24.04 Apptainer image (glibc 2.39) that contains Mojo but
bind-mounts the host's OpenMPI + CUDA at runtime:

```bash
cd /gscratch/aaplasma/embluhm/code/mojoxm
apptainer build --fakeroot mojo.sif mojo.def       # ~3 min, no GPU required
```

Rebuild only when bumping the Mojo nightly or changing `mojo.def`. The host
OpenMPI (`/sw/ompi/4.1.6-2`) and CUDA (`/sw/cuda/12.9.1`) are bind-mounted in
per-run via `--bind /sw --bind /gscratch` (handled by `klone-run`).

## Harmless noise to ignore

Two messages appear once per rank on every multi-rank run and can be
confidently ignored:

```
mca_base_component_repository_open: unable to open mca_btl_vader:
    libxpmem.so.0: cannot open shared object file: No such file or directory
    (ignored)
```
OpenMPI trying to load the XPMEM shared-memory transport; libxpmem isn't in
the container. OpenMPI falls back to `sm` / `self`. The trailing `(ignored)`
is from OpenMPI itself. Silence by bind-mounting xpmem into the container if
it bothers you, not worth it otherwise.

```
UCX ERROR  open(file_name=/proc/<PID>/fd/42 flags=0x0) failed: Permission denied
```
UCX's cross-process shared-memory optimization needs to walk a peer rank's
`/proc/<PID>/fd`, which Apptainer's security policy blocks. UCX falls back to
TCP. Silence with `OMPI_MCA_pml=ob1` / `OMPI_MCA_btl=self,tcp` or ignore.

Neither affects correctness — the MPI correctness test (`make test`) passes
bit-identically across rank counts despite both messages.

## Common pitfalls

- **Typo `PROFILE=WRAPPER=…`** instead of `PROFILE_WRAPPER=…`: bash parses
  this as one assignment setting `PROFILE` to the literal string
  `WRAPPER=/path/...`, so the wrapper never attaches. Look at the
  `===> mpirun …` echo line — if `[gpu-pin] apptainer exec` goes straight to
  apptainer with no wrapper in between, the var didn't set.
- **`NP=N` on a non-MPI driver**: silently ran one process before we added
  the warning; now prints a 3-line warning and still single-processes.
  Multi-rank DG solves live in `mpi_*` drivers; single-rank Euler/advection
  drivers cannot be parallelized this way without a new `mpi_euler_*` driver.
- **`CUDA_ERROR_NO_BINARY_FOR_GPU`**: you built on an A40 node and ran on an
  RTX6k (or vice versa). `klone-run` avoids this by compiling inside the
  same srun that runs the binary — never build from the login node or a
  previous allocation and keep the binary.
- **Queue stuck on "Resources"**: the GPU type you asked for is fully busy.
  Either wait, or re-dispatch with `CONSTRAINT='a40|rtx6k|…'` to pick up any
  idle GPU. `sinfo -p ckpt -t idle -o "%P %G %N"` shows what's free.
- **Long walltime with `TIME=…`**: default is 30 minutes. Bump for longer
  production runs; `ckpt` will preempt after your walltime regardless.
- **Writing output to `$HOME`**: home has small per-user quotas; VTU frame
  output and nsys traces blow past them quickly (a single 48³ advection run
  emits ~2.6 GB of VTU). `klone-run` runs with cwd at the project dir in
  `/gscratch`; keep it that way. Use `quota -s` to check current usage.

## Adding a new MPI driver

If you write `examples/mpi_euler_vortex.mojo` (or any new `mpi_*` driver),
klone-run picks it up automatically — the `mpi_*` prefix is the only
classification input, and the `Makefile` builds any `examples/<name>.mojo`
by name. `NEEDS_GPU` defaults to 1 for anything not in the
`mpi_hello|mpi_partition` allowlist; if your new driver truly doesn't touch
the GPU, add it to that case arm in `klone-run` so it doesn't waste a GPU
reservation.
