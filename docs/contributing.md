---
icon: lucide/git-pull-request
---

# Contributing

Welcome. Workflow notes for getting a change into mojoxm.

## Before you start

- Pick a focused, single-physics or single-pipeline change. The codebase
  rewards narrow PRs.
- Check the [Architecture overview](architecture/index.md#limitations-and-roadmap)
  for the current roadmap items. If your change overlaps with one of
  them, mention it in the PR description.

## First-time setup

Once per clone, wire up the pre-commit hook so staged `.mojo` files
get format-checked automatically before each commit:

```bash
make install-hooks
```

This sets `core.hooksPath` to `scripts/git-hooks/`, which contains a
`pre-commit` script that runs `mojo format` on every staged
`.mojo` file (validating staged content directly, so partial-stage
hunks are checked correctly).  The hook **never modifies files** —
if any are out of conformance it lists them and aborts the commit.
Run `make format` to fix.

## Local checks

The pre-push gate is:

```bash
make pre-push          # ~50 s cached
```

This chains:

| Step                       | Time   | What it catches                                 |
| -------------------------- | ------ | ----------------------------------------------- |
| `format-check`             | ~3 s   | Mojo formatter drift                            |
| `profile-summary-test`     | ~0.2 s | Self-test on the cross-bench summary parser     |
| `smoke`                    | ~45 s  | `test-utils + test-quick + bench-quick`         |
| `test-vtu-meshio`          | ~1 s   | Meshio roundtrip + VTK_LAGRANGE_* spec compliance |

Run this before every push. If it's red locally, CI will be red too.

## Coding conventions

The codebase has a few firm conventions, captured in skill notes:

- **`def`, never `fn`.** All Mojo function/method definitions use `def`.
  `fn` is deprecated in modern Mojo.
- **Keyword args for non-trivial calls.** For \(\geq 3\) arguments,
  prefer `f(x=1, y=2)`. Trivial conversions like `Float32(0.5)` stay
  positional.
- **GPU-only.** Don't add CPU drivers, CPU reference physics, or
  GPU-vs-CPU diff harnesses. Validate with GPU self-consistency +
  analytic benchmarks instead.
- **No `make -j`.** Mojo is already multi-threaded per file. `make -j4`
  is measurably slower than serial.
- **Profile before refactoring for GPU coalescing.** L2 cache absorbs
  much of the nominally-uncoalesced cost. Uncoalesced ≠ memory-bound.
  See the limiter density pre-pack story in
  [Reference · Limiter](reference/limiter.md#cost-share-by-p).

## Adding a new physics

See [Architecture · Physics trait](architecture/physics-trait.md#how-to-add-a-new-physics)
for the step-by-step:

1. Create `src/<physics>.mojo` satisfying the trait
2. Cover every supported `bc_type` in `boundary_flux`
3. Add a 3D constant-state test driver
4. Add benches for any analytic solution your physics admits
5. Optionally add a 2D-GPU module
6. Wire the new tests/benches into the relevant `make` aggregator

Every BC dispatch arm in every physics must have at least one direct
benchmark gate. The existing per-physics aggregators (`make bench-mhd`,
`bench-euler`, etc.) enforce this by file-listing rather than by
runtime check.

## Adding a new benchmark

Bench files live under `benchmarks/` and follow the pattern:

```mojo
def main():
    # 1. Construct solver + IC for a problem with a known analytic answer
    # 2. Run the SSPRK3 loop to T_FINAL
    # 3. Compute analytic q at T_FINAL on the same mesh
    # 4. Compute relative L2 (or whatever invariant your problem admits)
    # 5. expect[Float32](err < threshold, descriptive message)
    # 6. print(...) the measured numbers for diff-ability
```

Threshold sizing: **~1.4–3× the empirical error floor**. Tight enough to
catch any meaningful regression, loose enough to absorb run-to-run
floating-point variation.

If your bench has a closed-form rate (e.g., a smooth-wave problem at
multiple polynomial orders), prefer a `log₂(e_N / e_{2N})` rate gate
rather than an absolute-L2 sentinel. See `bench_advection_translation_2d_p5`
for the reference template.

## Adding a Makefile target

New benches need 5 wirings (the generic `$(BENCH_DRIVERS): %:
benchmarks/%.mojo ...` rule handles compilation, so you don't write a
per-bench build rule — just **list** the binary name):

1. **`BENCH_DRIVERS = ...`** at the top of the Makefile — append the
   binary name (e.g. `bench_my_thing`).  This is what gets the
   compile rule.
2. **Per-bench *run* rule**:
   ```make
   bench-my-thing: bench_my_thing
       ./bench_my_thing
   ```
3. **Per-bench *profile* rule** (so `make profile-bench-my-thing`
   regenerates the cached nsys baseline):
   ```make
   profile-bench-my-thing: bench_my_thing
       @bin=bench_my_thing; $(PROFILE_BIN)
   ```
4. **Aggregators** that should include the new bench:
    - the relevant per-physics aggregator (`bench-euler` /
      `bench-mhd` / `bench-maxwell` / etc.)
    - any cross-cutting aggregator that applies (`bench-rates` /
      `bench-bcs` / `bench-shocks` / `bench-p5`)
    - `bench-all`
    - the parallel `profile-bench-all`
5. **Update the aggregator's count line** (e.g.
   `=== bench-maxwell: 26 Maxwell-focused gates PASSED ===` if your
   addition makes it 27).

The Makefile is the source of truth for what runs in CI. If your bench
isn't in `bench-all`, it isn't gating anything.

Then run a baseline profile so the regression-detection pipeline
has a checked-in `.kern.txt`:

```bash
make profile-bench-my-thing
git add benchmarks/profile_reports/bench_my_thing.kern.txt
```

## Documentation

If your change touches user-facing behaviour, update the docs site
(this site). Run a local preview:

```bash
make docs-serve         # http://localhost:8000
```

Or build and inspect:

```bash
make docs-build
xdg-open site/index.html
```

Docs are written in CommonMark with the [Zensical](https://zensical.org/)
extensions enabled — admonitions, content tabs, code annotations,
mathjax. See [`docs/getting-started/your-first-simulation.md`](getting-started/your-first-simulation.md)
for a representative shape.
