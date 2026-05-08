---
icon: lucide/fence
---

# Boundary conditions

mojoxm has four BC kinds, defined as integer constants in
`src/boundary.mojo`:

| Constant      | Value | Default behaviour (overridable per physics)                  |
| ------------- | ----- | ------------------------------------------------------------ |
| `BC_PERIODIC` | 0     | Wraparound — handled by the mesh, not by physics             |
| `BC_WALL`     | 1     | Slip wall (no-penetration). PEC for Maxwell.                 |
| `BC_OUTFLOW`  | 2     | Transmissive (one-way absorber)                              |
| `BC_INFLOW`   | 3     | Riemann against a prescribed inflow ghost                    |

Periodic BCs are baked into the mesh: a face flagged `BC_PERIODIC` has its
neighbour element set by the periodic-wrap rule and goes through
`numerical_flux`, not `boundary_flux`. The other three kinds always
dispatch into `boundary_flux(q_int, bc_type, n)`.

## How dispatch works

```mojo
struct Euler:
    def boundary_flux(
        self, q_int, bc_type, nx, ny, nz, flux,
    ) -> Float32:
        if bc_type == BC_WALL:
            return self._wall_flux(q_int, nx, ny, nz, flux)
        elif bc_type == BC_OUTFLOW:
            return self._outflow_flux(q_int, nx, ny, nz, flux)
        elif bc_type == BC_INFLOW:
            return self._inflow_flux(q_int, nx, ny, nz, flux)
        else:
            return Float32(0)        # unreachable
```

Every physics module fills out this dispatch arm. Missing a `bc_type`
constant in a physics manifests as a silent fall-through — which is why
**every BC dispatch arm in every physics has at least one direct
benchmark gate**:

| Physics                | BC_WALL gate                               | BC_OUTFLOW gate                            | BC_INFLOW gate                                |
| ---------------------- | ------------------------------------------ | ------------------------------------------ | --------------------------------------------- |
| Advection              | (BC_WALL not used)                         | `bench_advection_outflow_{2d,3d}`          | `bench_advection_inflow_{2d,3d}`              |
| Euler                  | (slip wall used in Sod)                    | `bench_euler_inflow_*` covers + outflow    | `bench_euler_inflow_{2d,3d}`                  |
| ShallowWater           | dam break gates                            | `bench_shallow_water_inflow_*`             | `bench_shallow_water_inflow_{2d,3d}`          |
| Maxwell                | cavity gates (PEC)                         | `bench_maxwell_outflow_{2d,3d}`            | `bench_maxwell_inflow_{2d,3d}`                |
| IdealMHD               | `bench_mhd_wall_{2d,3d}{,_glm}`            | `bench_mhd_inflow_{2d,3d}` (+ outflow)     | `bench_mhd_inflow_{2d,3d}{,_glm}`             |
| FiveMomentTwoFluid     | `bench_two_fluid_walls_3d_p{2..5}`         | `bench_two_fluid_outflow_3d`               | `bench_two_fluid_inflow_3d`                   |

## How a driver declares BCs

```mojo
var bcs = BoundaryConditions(
    x_minus=BC_INFLOW,  x_plus=BC_OUTFLOW,
    y_minus=BC_WALL,    y_plus=BC_WALL,
    z_minus=BC_WALL,    z_plus=BC_WALL,
)
var mesh = Mesh[P].build_with_bcs_3d(NX, NY, NZ, LX, LY, LZ, bcs)
```

The mesh builder allocates BC-side faces as separate face slots and
routes them through the physics's `boundary_flux` instead of the interior
`numerical_flux`. The 2D mesh path takes the same struct (with only `x_*,
y_*` populated).

## Physics-specific notes

### `BC_WALL`

| Physics            | Concrete behaviour                                                              |
| ------------------ | ------------------------------------------------------------------------------- |
| Advection          | Reflect: \(\hat{F} = 0\)                                                        |
| Euler              | Slip wall: mirror normal velocity, identical pressure                           |
| ShallowWater       | Slip wall: mirror normal velocity, identical h                                  |
| Maxwell            | PEC: tangential \(\mathbf{E} = 0\), normal \(\mathbf{B} = 0\)                  |
| IdealMHD           | Slip wall: mirror normal velocity, identical pressure / B                      |
| FiveMomentTwoFluid | Per-block: slip on each fluid, PEC on the EM block                              |

### `BC_OUTFLOW`

Implemented identically across physics: ghost = interior; flux is the
pure upwind / Rusanov flux against the extrapolated state. For Euler
this is exact for supersonic outflow and slightly dissipative for
subsonic.

### `BC_INFLOW`

The driver provides an `inflow_q` ghost state — an `NC`-vector
representing the desired upstream conditions. The boundary flux is
the Riemann flux between the interior state and `inflow_q`. For
Maxwell this is used for analytic plane-wave injection; for Euler
for prescribed channel inflow; for the two-fluid module for charged
beam injection.

## Periodic-wrap detail

Periodic BCs aren't really a "BC dispatch arm" — they're baked into the
mesh by the host-side periodic-wrap rule:

```text
elem_neighbour_across_face[e, f] = wrap(e + face_to_offset[f])
```

Translation invariance of the Kuhn decomposition makes this dictionary-
free: the offset table is fixed at build time and doesn't depend on `e`.
The `numerical_flux` kernel then runs across the periodic wrap as if it
were any other interior face — no special case in the inner loop.
