---
icon: lucide/package
---

# Installation

mojoxm uses [Pixi](https://pixi.sh/) to manage its conda environment. Pixi
pulls Mojo from the Modular `max-nightly` channel and the Python helpers
(`meshio` for VTU readback, `scipy` for the dashboard scripts) from
`conda-forge`. The provided `pixi.toml` is the source of truth — don't
shadow it with `conda create` if you can avoid it.

## 1. Install Pixi

=== "macOS / Linux"

    ```bash
    curl -fsSL https://pixi.sh/install.sh | bash
    ```

=== "Homebrew"

    ```bash
    brew install pixi
    ```

=== "Windows"

    ```powershell
    iwr -useb https://pixi.sh/install.ps1 | iex
    ```

Restart your shell so `pixi` lands on `$PATH`.

## 2. Clone and bootstrap the environment

```bash
git clone https://github.com/EvanBluhm/mojoxm
cd mojoxm
pixi install        # materialises .pixi/envs/default
```

`pixi install` is idempotent — re-run it whenever `pyproject.toml` or
`pixi.lock` changes. The environment lives in `.pixi/` and is
`.gitignore`d.

## 3. Sanity-check the Mojo toolchain

```bash
pixi run mojo --version
```

You should see something like `mojo 0.7.x (...)`. The `Makefile` defaults
`MOJO` to the pixi-resolved binary, so once `pixi install` succeeds you
can use `make` directly.

## 4. Build everything (optional but recommended)

```bash
make all
```

This compiles every example driver, benchmark, and unit test. On an RTX
3090 / WSL2 box it takes ~2 minutes.

!!! warning "Do not pass `-j`"

    Mojo's compiler is already multi-threaded per file. Empirically `make -j4`
    is **slower** than serial — measured 18s vs 12s for a 4-bench batch.
    Just `make` does the right thing.

## 5. Run the smoke gate

```bash
make smoke
```

Runs `test-quick` + `bench-quick` (~45 s cached). If this passes, your
toolchain is healthy.

## Optional: docs site (this site)

The documentation you are reading is built with [Zensical](https://zensical.org/),
a static site generator written in Rust + Python:

```bash
pip install zensical
zensical serve            # http://localhost:8000
```

Or, in a Pixi-managed venv:

```bash
pixi run pip install zensical
pixi run zensical serve
```

A persistent `make docs-serve` / `make docs-build` target is wired up too:

```bash
make docs-serve     # live-reload preview
make docs-build     # build to ./site/
```

## Troubleshooting

??? failure "`mojo: command not found`"

    Pixi installs Mojo inside `.pixi/envs/default/bin`. Either prefix every
    command with `pixi run`, or activate the env: `pixi shell`.

??? failure "`CUDA_ERROR_NO_BINARY_FOR_GPU`"

    Most often a driver / toolkit mismatch on a cluster node. If you're on
    UW Klone, see the `running-on-klone` skill — `scripts/klone-run` wraps
    the SLURM + apptainer dance.

??? failure "Pre-push hook fails on `format-check`"

    Run `pixi run mojo format src/ examples/ benchmarks/ test/` to fix
    formatting in place. The `format_check.sh` helper script is also
    available.
