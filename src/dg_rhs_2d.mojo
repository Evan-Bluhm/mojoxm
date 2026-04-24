# ======================================================================
# Host-side 2D DG right-hand side evaluator (physics-generic)
# ======================================================================
#
# Evaluates one RK stage's rhs for a 2D hyperbolic conservation law on
# a `LocalMesh2D[P]` using a `ReferenceElement2D[P]`.  Dispatches flux
# evaluation through a `Physics2D` trait so the same pipeline handles
# scalar advection today and Euler / shallow water / MHD later.  Runs
# entirely on CPU in Float64 -- intended for correctness prototyping
# before the 2D GPU solver lands (see project_2d_triangles_scope.md).
#
# Weak-form DG with nodal Lagrange collocation; at each owned node
#
#   rhs[e, i, c] = vol_c - inv_2A * face_c
#
#   vol_c        = sum_j D_ref[k, i, j] * (invJ[k, :] . F(q_j))_k
#   face_c       = sum_faces sign * face_len * Lift_ref[lf, i, r] * fstar_m
#
# `fstar` comes from `physics.numerical_flux(q_l, q_r, nx, ny, ...)` at
# each face node; `F(q_j)` from `physics.internal_flux(q, ...)`.  Sign
# convention: rhs returns dq/dt (semi-discrete time derivative), so an
# SSPRK3 driver integrates as q_new = a * q_a + b * q_b + cc * dt * rhs.
# ======================================================================

from std.math import sqrt
from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.boundary import BC_INTERIOR, BC_WALL, BC_OUTFLOW, BC_INFLOW


# ----------------------------------------------------------------------
# Physics2D trait
# ----------------------------------------------------------------------
# Concrete 2D physics types (Advection2D for now; Euler2D / ShallowWater2D
# later) implement these three methods.  Keeps the same overall shape
# as the 3D `Physics` trait in src/solver.mojo, minus the z-component.
# ----------------------------------------------------------------------

trait Physics2D(Copyable, Movable, ImplicitlyDestructible):
    comptime NUM_COMPONENTS: Int

    def internal_flux(
        self,
        q:    UnsafePointer[Float64, MutAnyOrigin],
        flux: UnsafePointer[Float64, MutAnyOrigin],
    ) -> Float64:
        """Fills flux[d * NC + c] for d in {0=x, 1=y}, c in range(NC).
        Returns max |wave speed| (for CFL estimation)."""
        ...

    def numerical_flux(
        self,
        q_l:  UnsafePointer[Float64, MutAnyOrigin],
        q_r:  UnsafePointer[Float64, MutAnyOrigin],
        nx: Float64, ny: Float64,
        flux: UnsafePointer[Float64, MutAnyOrigin],
    ) -> Float64:
        """Fills flux[c] for c in range(NC).  Returns max |wave speed|
        at the interface (Rusanov / Lax-Friedrichs dissipation needs
        this)."""
        ...

    def boundary_flux(
        self,
        q_int: UnsafePointer[Float64, MutAnyOrigin],
        bc_type: Int32,
        nx: Float64, ny: Float64,
        flux: UnsafePointer[Float64, MutAnyOrigin],
    ) -> Float64:
        """Face flux on a non-periodic boundary edge.  `q_int` is the
        interior element's state at the face node; the (non-existent)
        ghost side is derived from `bc_type` (BC_WALL, BC_OUTFLOW, ...
        from src.boundary).  The outward normal (nx, ny) points from
        interior to ghost.  Fills `flux[c]` for c in range(NC); returns
        max |wave speed|."""
        ...

    def source_term(
        self,
        q: UnsafePointer[Float64, MutAnyOrigin],
        x: Float64, y: Float64,
        source_out: UnsafePointer[Float64, MutAnyOrigin],
    ):
        """Pointwise nodal source S(q, x, y); physics with no source
        zero-fills."""
        ...


# ----------------------------------------------------------------------
# Advection2D: scalar linear advection
# ----------------------------------------------------------------------

struct Advection2D(Physics2D, ImplicitlyCopyable, Movable):
    comptime NUM_COMPONENTS = 1

    var vx: Float64
    var vy: Float64
    # Inflow state (used only when a boundary face has bc_type ==
    # BC_INFLOW).  Default 0 so pre-BC_INFLOW drivers work unchanged.
    var inflow_q: Float64

    def __init__(
        out self,
        vx: Float64, vy: Float64,
        inflow_q: Float64 = 0.0,
    ):
        self.vx = vx
        self.vy = vy
        self.inflow_q = inflow_q

    def internal_flux(
        self,
        q:    UnsafePointer[Float64, MutAnyOrigin],
        flux: UnsafePointer[Float64, MutAnyOrigin],
    ) -> Float64:
        flux[0] = self.vx * q[0]  # d=0 (x)
        flux[1] = self.vy * q[0]  # d=1 (y)
        return sqrt(self.vx * self.vx + self.vy * self.vy)

    def numerical_flux(
        self,
        q_l:  UnsafePointer[Float64, MutAnyOrigin],
        q_r:  UnsafePointer[Float64, MutAnyOrigin],
        nx: Float64, ny: Float64,
        flux: UnsafePointer[Float64, MutAnyOrigin],
    ) -> Float64:
        var vn = self.vx * nx + self.vy * ny
        var abs_vn = vn if vn >= 0.0 else -vn
        if vn >= 0.0:
            flux[0] = vn * q_l[0]
        else:
            flux[0] = vn * q_r[0]
        return abs_vn

    def boundary_flux(
        self,
        q_int: UnsafePointer[Float64, MutAnyOrigin],
        bc_type: Int32,
        nx: Float64, ny: Float64,
        flux: UnsafePointer[Float64, MutAnyOrigin],
    ) -> Float64:
        var vn = self.vx * nx + self.vy * ny
        var abs_vn = vn if vn >= 0.0 else -vn
        # For advection, the physically correct boundary flux depends
        # on whether the characteristic is outgoing (vn > 0) or incoming
        # (vn < 0) relative to the outward normal:
        #   * Outgoing: upwind = interior; all BC kinds agree.
        #   * Incoming: upwind = ghost.  BC_WALL / BC_OUTFLOW use
        #     a zero-Dirichlet ghost (nothing enters); BC_INFLOW uses
        #     the `inflow_q` carried on the physics instance (set by
        #     the driver's Advection2D(..., inflow_q=...) constructor).
        if vn >= 0.0:
            flux[0] = vn * q_int[0]
        else:
            if bc_type == BC_INFLOW:
                flux[0] = vn * self.inflow_q
            else:
                flux[0] = 0.0
        return abs_vn

    def source_term(
        self,
        q: UnsafePointer[Float64, MutAnyOrigin],
        x: Float64, y: Float64,
        source_out: UnsafePointer[Float64, MutAnyOrigin],
    ):
        source_out[0] = 0.0


# ----------------------------------------------------------------------
# Euler2D: compressible gas dynamics (Rusanov / Lax-Friedrichs flux)
# ----------------------------------------------------------------------
# State: (rho, rho*u, rho*v, E).  Ideal gas EOS p = (gamma - 1) (E - KE).
# Numerical flux is plain Rusanov -- robust, monotone, a good default
# for prototyping.  A full 2D Euler suite (Roe / HLLE / HLLEC) can
# parallel the 3D module if the demand ever arrives.

struct Euler2D(Physics2D, ImplicitlyCopyable, Movable):
    comptime NUM_COMPONENTS = 4

    var gamma: Float64
    var min_density: Float64
    var min_pressure: Float64
    # BC_INFLOW ghost state in conservative variables
    # (rho, rho*u, rho*v, E).  Defaults zero so existing drivers work.
    var inflow_rho: Float64
    var inflow_rhou: Float64
    var inflow_rhov: Float64
    var inflow_E: Float64

    def __init__(
        out self,
        gamma: Float64,
        min_density: Float64,
        min_pressure: Float64,
        inflow_rho: Float64 = 0.0,
        inflow_rhou: Float64 = 0.0,
        inflow_rhov: Float64 = 0.0,
        inflow_E: Float64 = 0.0,
    ):
        self.gamma = gamma
        self.min_density = min_density
        self.min_pressure = min_pressure
        self.inflow_rho = inflow_rho
        self.inflow_rhou = inflow_rhou
        self.inflow_rhov = inflow_rhov
        self.inflow_E = inflow_E

    def internal_flux(
        self,
        q:    UnsafePointer[Float64, MutAnyOrigin],
        flux: UnsafePointer[Float64, MutAnyOrigin],
    ) -> Float64:
        var rho = q[0]
        if rho < self.min_density:
            rho = self.min_density
        var mx = q[1]
        var my = q[2]
        var E = q[3]
        var u = mx / rho
        var v = my / rho
        var ke = 0.5 * rho * (u * u + v * v)
        var p = (self.gamma - 1.0) * (E - ke)
        if p < self.min_pressure:
            p = self.min_pressure
        # x-direction
        flux[0] = mx
        flux[1] = mx * u + p
        flux[2] = mx * v
        flux[3] = u * (E + p)
        # y-direction
        flux[4] = my
        flux[5] = my * u
        flux[6] = my * v + p
        flux[7] = v * (E + p)
        var c = sqrt(self.gamma * p / rho)
        var vmag = sqrt(u * u + v * v)
        return vmag + c

    def numerical_flux(
        self,
        q_l:  UnsafePointer[Float64, MutAnyOrigin],
        q_r:  UnsafePointer[Float64, MutAnyOrigin],
        nx: Float64, ny: Float64,
        flux: UnsafePointer[Float64, MutAnyOrigin],
    ) -> Float64:
        # Rusanov / Lax-Friedrichs:  F* = 0.5 (F_L.n + F_R.n) - 0.5 alpha (q_R - q_L)
        # alpha = max(|v.n| + c) over the two sides.
        var f_l_buf = InlineArray[Float64, 8](fill=0.0)
        var f_r_buf = InlineArray[Float64, 8](fill=0.0)
        var f_l = rebind[UnsafePointer[Float64, MutAnyOrigin]](
            f_l_buf.unsafe_ptr()
        )
        var f_r = rebind[UnsafePointer[Float64, MutAnyOrigin]](
            f_r_buf.unsafe_ptr()
        )
        var speed_l = self.internal_flux(q_l, f_l)
        var speed_r = self.internal_flux(q_r, f_r)
        var alpha = speed_l if speed_l > speed_r else speed_r

        for c in range(4):
            var Fn_l = f_l[0 * 4 + c] * nx + f_l[1 * 4 + c] * ny
            var Fn_r = f_r[0 * 4 + c] * nx + f_r[1 * 4 + c] * ny
            flux[c] = 0.5 * (Fn_l + Fn_r) - 0.5 * alpha * (q_r[c] - q_l[c])
        return alpha

    def boundary_flux(
        self,
        q_int: UnsafePointer[Float64, MutAnyOrigin],
        bc_type: Int32,
        nx: Float64, ny: Float64,
        flux: UnsafePointer[Float64, MutAnyOrigin],
    ) -> Float64:
        # Build a ghost state q_ghost based on bc_type, then call the
        # interior numerical flux with (q_int, q_ghost).  Mirrors the
        # 3D Euler convention.
        var q_g_buf = InlineArray[Float64, 4](fill=0.0)
        if bc_type == BC_WALL:
            # Reflect normal momentum: q_ghost has (rho, m_t, -m_n, E).
            var mx = q_int[1]
            var my = q_int[2]
            var m_n = mx * nx + my * ny
            q_g_buf[0] = q_int[0]
            q_g_buf[1] = mx - 2.0 * m_n * nx
            q_g_buf[2] = my - 2.0 * m_n * ny
            q_g_buf[3] = q_int[3]
        elif bc_type == BC_INFLOW:
            # Dirichlet inflow: ghost = user-specified state carried on
            # the physics instance.
            q_g_buf[0] = self.inflow_rho
            q_g_buf[1] = self.inflow_rhou
            q_g_buf[2] = self.inflow_rhov
            q_g_buf[3] = self.inflow_E
        else:
            # BC_OUTFLOW (default): zero-gradient ghost.
            for c in range(4):
                q_g_buf[c] = q_int[c]
        var q_g = rebind[UnsafePointer[Float64, MutAnyOrigin]](
            q_g_buf.unsafe_ptr()
        )
        return self.numerical_flux(q_int, q_g, nx, ny, flux)

    def source_term(
        self,
        q: UnsafePointer[Float64, MutAnyOrigin],
        x: Float64, y: Float64,
        source_out: UnsafePointer[Float64, MutAnyOrigin],
    ):
        for c in range(4):
            source_out[c] = 0.0


# ----------------------------------------------------------------------
# ShallowWater2D: Saint-Venant shallow-water equations
# ----------------------------------------------------------------------
# State: (h, h*u, h*v) where h is water column depth and (u, v) is the
# depth-averaged horizontal velocity.  "Pressure" analog p = g h^2 / 2.
# Flat bed (no source term).  Rusanov numerical flux -- same pattern
# as Euler2D, just a simpler 3-component state.

struct ShallowWater2D(Physics2D, ImplicitlyCopyable, Movable):
    comptime NUM_COMPONENTS = 3

    var g: Float64            # gravitational acceleration
    var min_h: Float64        # depth floor (avoids divide-by-zero at dry patches)
    # BC_INFLOW ghost state (h, h*u, h*v).  Defaults zero.
    var inflow_h: Float64
    var inflow_hu: Float64
    var inflow_hv: Float64

    def __init__(
        out self,
        g: Float64,
        min_h: Float64,
        inflow_h: Float64 = 0.0,
        inflow_hu: Float64 = 0.0,
        inflow_hv: Float64 = 0.0,
    ):
        self.g = g
        self.min_h = min_h
        self.inflow_h = inflow_h
        self.inflow_hu = inflow_hu
        self.inflow_hv = inflow_hv

    def internal_flux(
        self,
        q:    UnsafePointer[Float64, MutAnyOrigin],
        flux: UnsafePointer[Float64, MutAnyOrigin],
    ) -> Float64:
        var h = q[0]
        if h < self.min_h:
            h = self.min_h
        var mx = q[1]
        var my = q[2]
        var u = mx / h
        var v = my / h
        var p = 0.5 * self.g * h * h
        # x-direction
        flux[0] = mx
        flux[1] = mx * u + p
        flux[2] = mx * v
        # y-direction
        flux[3] = my
        flux[4] = my * u
        flux[5] = my * v + p
        var c = sqrt(self.g * h)
        var vmag = sqrt(u * u + v * v)
        return vmag + c

    def numerical_flux(
        self,
        q_l:  UnsafePointer[Float64, MutAnyOrigin],
        q_r:  UnsafePointer[Float64, MutAnyOrigin],
        nx: Float64, ny: Float64,
        flux: UnsafePointer[Float64, MutAnyOrigin],
    ) -> Float64:
        var f_l_buf = InlineArray[Float64, 6](fill=0.0)
        var f_r_buf = InlineArray[Float64, 6](fill=0.0)
        var f_l = rebind[UnsafePointer[Float64, MutAnyOrigin]](
            f_l_buf.unsafe_ptr()
        )
        var f_r = rebind[UnsafePointer[Float64, MutAnyOrigin]](
            f_r_buf.unsafe_ptr()
        )
        var speed_l = self.internal_flux(q_l, f_l)
        var speed_r = self.internal_flux(q_r, f_r)
        var alpha = speed_l if speed_l > speed_r else speed_r

        for c in range(3):
            var Fn_l = f_l[0 * 3 + c] * nx + f_l[1 * 3 + c] * ny
            var Fn_r = f_r[0 * 3 + c] * nx + f_r[1 * 3 + c] * ny
            flux[c] = 0.5 * (Fn_l + Fn_r) - 0.5 * alpha * (q_r[c] - q_l[c])
        return alpha

    def boundary_flux(
        self,
        q_int: UnsafePointer[Float64, MutAnyOrigin],
        bc_type: Int32,
        nx: Float64, ny: Float64,
        flux: UnsafePointer[Float64, MutAnyOrigin],
    ) -> Float64:
        var q_g_buf = InlineArray[Float64, 3](fill=0.0)
        if bc_type == BC_WALL:
            # Reflect normal momentum (depth unchanged).
            var mx = q_int[1]
            var my = q_int[2]
            var m_n = mx * nx + my * ny
            q_g_buf[0] = q_int[0]
            q_g_buf[1] = mx - 2.0 * m_n * nx
            q_g_buf[2] = my - 2.0 * m_n * ny
        elif bc_type == BC_INFLOW:
            q_g_buf[0] = self.inflow_h
            q_g_buf[1] = self.inflow_hu
            q_g_buf[2] = self.inflow_hv
        else:
            for c in range(3):
                q_g_buf[c] = q_int[c]
        var q_g = rebind[UnsafePointer[Float64, MutAnyOrigin]](
            q_g_buf.unsafe_ptr()
        )
        return self.numerical_flux(q_int, q_g, nx, ny, flux)

    def source_term(
        self,
        q: UnsafePointer[Float64, MutAnyOrigin],
        x: Float64, y: Float64,
        source_out: UnsafePointer[Float64, MutAnyOrigin],
    ):
        for c in range(3):
            source_out[c] = 0.0


# ----------------------------------------------------------------------
# IdealMHD2D: ideal magnetohydrodynamics (6-component)
# ----------------------------------------------------------------------
# State: (rho, rho*u, rho*v, Bx, By, E).  No z-momentum / Bz; flat-plane
# magnetic-field configuration.  No GLM divergence cleaning here -- a
# pure constraint-transport equilibrium is preserved if the IC is
# divergence-free, but the scheme doesn't self-correct violations.
# For routine test problems (Alfven wave, OT vortex reduced to 2D) this
# is usually acceptable; for stiff shock cases with strong compressions
# a Dedner-style GLM psi could be added as a 7th component following
# the 3D MHD module's pattern.

@fieldwise_init
struct IdealMHD2D(Physics2D, ImplicitlyCopyable, Movable):
    comptime NUM_COMPONENTS = 6

    var gamma: Float64
    var min_density: Float64
    var min_pressure: Float64

    def _p_thermal(self, q: UnsafePointer[Float64, MutAnyOrigin]) -> Float64:
        """Thermal pressure from the state: p = (g-1) (E - KE - MP)."""
        var rho = q[0]
        if rho < self.min_density:
            rho = self.min_density
        var mx = q[1]
        var my = q[2]
        var Bx = q[3]
        var By = q[4]
        var E  = q[5]
        var ke = 0.5 * (mx * mx + my * my) / rho
        var mp = 0.5 * (Bx * Bx + By * By)
        var p  = (self.gamma - 1.0) * (E - ke - mp)
        if p < self.min_pressure:
            p = self.min_pressure
        return p

    def internal_flux(
        self,
        q:    UnsafePointer[Float64, MutAnyOrigin],
        flux: UnsafePointer[Float64, MutAnyOrigin],
    ) -> Float64:
        var rho = q[0]
        if rho < self.min_density:
            rho = self.min_density
        var mx = q[1]
        var my = q[2]
        var Bx = q[3]
        var By = q[4]
        var E  = q[5]
        var u = mx / rho
        var v = my / rho
        var p = self._p_thermal(q)
        var BB = Bx * Bx + By * By
        var pstar = p + 0.5 * BB           # total (thermal + magnetic) pressure

        # x-flux
        flux[0] = mx
        flux[1] = mx * u + pstar - Bx * Bx
        flux[2] = mx * v         - Bx * By
        flux[3] = 0.0
        flux[4] = u * By - v * Bx
        flux[5] = (E + pstar) * u - Bx * (u * Bx + v * By)
        # y-flux
        flux[6]  = my
        flux[7]  = my * u         - By * Bx
        flux[8]  = my * v + pstar - By * By
        flux[9]  = v * Bx - u * By
        flux[10] = 0.0
        flux[11] = (E + pstar) * v - By * (u * Bx + v * By)

        # Fast magnetosonic speed as the wave bound.
        var cs2 = self.gamma * p / rho             # sound speed^2
        var ca2 = BB / rho                          # Alfven speed^2 (full)
        var s = cs2 + ca2
        var disc = s * s - 4.0 * cs2 * (Bx * Bx) / rho
        if disc < 0.0: disc = 0.0
        var cf2 = 0.5 * (s + sqrt(disc))
        var cf = sqrt(cf2)
        var vmag = sqrt(u * u + v * v)
        return vmag + cf

    def numerical_flux(
        self,
        q_l:  UnsafePointer[Float64, MutAnyOrigin],
        q_r:  UnsafePointer[Float64, MutAnyOrigin],
        nx: Float64, ny: Float64,
        flux: UnsafePointer[Float64, MutAnyOrigin],
    ) -> Float64:
        var f_l_buf = InlineArray[Float64, 12](fill=0.0)
        var f_r_buf = InlineArray[Float64, 12](fill=0.0)
        var f_l = rebind[UnsafePointer[Float64, MutAnyOrigin]](
            f_l_buf.unsafe_ptr()
        )
        var f_r = rebind[UnsafePointer[Float64, MutAnyOrigin]](
            f_r_buf.unsafe_ptr()
        )
        var sl = self.internal_flux(q_l, f_l)
        var sr = self.internal_flux(q_r, f_r)
        var alpha = sl if sl > sr else sr

        for c in range(6):
            var Fn_l = f_l[0 * 6 + c] * nx + f_l[1 * 6 + c] * ny
            var Fn_r = f_r[0 * 6 + c] * nx + f_r[1 * 6 + c] * ny
            flux[c] = 0.5 * (Fn_l + Fn_r) - 0.5 * alpha * (q_r[c] - q_l[c])
        return alpha

    def boundary_flux(
        self,
        q_int: UnsafePointer[Float64, MutAnyOrigin],
        bc_type: Int32,
        nx: Float64, ny: Float64,
        flux: UnsafePointer[Float64, MutAnyOrigin],
    ) -> Float64:
        var q_g_buf = InlineArray[Float64, 6](fill=0.0)
        if bc_type == BC_WALL:
            # Reflect normal momentum AND normal B (standard MHD slip
            # wall / perfectly-conducting wall).
            var mx = q_int[1]
            var my = q_int[2]
            var Bx = q_int[3]
            var By = q_int[4]
            var m_n = mx * nx + my * ny
            var B_n = Bx * nx + By * ny
            q_g_buf[0] = q_int[0]
            q_g_buf[1] = mx - 2.0 * m_n * nx
            q_g_buf[2] = my - 2.0 * m_n * ny
            q_g_buf[3] = Bx - 2.0 * B_n * nx
            q_g_buf[4] = By - 2.0 * B_n * ny
            q_g_buf[5] = q_int[5]
        else:
            # BC_OUTFLOW / BC_INTERIOR / BC_INFLOW (no inflow state
            # plumbed through for MHD in this minimal module): zero-
            # gradient ghost.
            for c in range(6):
                q_g_buf[c] = q_int[c]
        var q_g = rebind[UnsafePointer[Float64, MutAnyOrigin]](
            q_g_buf.unsafe_ptr()
        )
        return self.numerical_flux(q_int, q_g, nx, ny, flux)

    def source_term(
        self,
        q: UnsafePointer[Float64, MutAnyOrigin],
        x: Float64, y: Float64,
        source_out: UnsafePointer[Float64, MutAnyOrigin],
    ):
        for c in range(6):
            source_out[c] = 0.0


# ----------------------------------------------------------------------
# Physics-generic 2D DG rhs
# ----------------------------------------------------------------------

def dg_rhs_2d[P: Int, PhysT: Physics2D](
    mesh: LocalMesh2D[P],
    re: ReferenceElement2D[P],
    physics: PhysT,
    q_in: List[Float64],
    mut rhs: List[Float64],
) raises:
    comptime NC = PhysT.NUM_COMPONENTS
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    var n_total = mesh.num_elements * NP * NC
    if len(q_in) != n_total:
        raise Error("dg_rhs_2d: q_in size mismatch")
    if len(rhs) != n_total:
        raise Error("dg_rhs_2d: rhs size mismatch")
    for k in range(n_total):
        rhs[k] = 0.0

    var q_ptr = rebind[UnsafePointer[Float64, MutAnyOrigin]](
        q_in.unsafe_ptr()
    )

    # 1) Per-face numerical flux at every face-local slot.
    var fstar_buf = List[Float64]()
    for _ in range(mesh.num_faces * NFP * NC):
        fstar_buf.append(0.0)
    var fstar_ptr = rebind[UnsafePointer[Float64, MutAnyOrigin]](
        fstar_buf.unsafe_ptr()
    )

    for fid in range(mesh.num_faces):
        var e_l = Int(mesh.face_elem[fid * 2 + 0])
        var e_r = Int(mesh.face_elem[fid * 2 + 1])
        var nx = mesh.face_normal[fid * 2 + 0]
        var ny = mesh.face_normal[fid * 2 + 1]
        var bc_type = mesh.face_bc_type[fid]
        for m in range(NFP):
            var n_l = Int(mesh.face_elem_node[(fid * 2 + 0) * NFP + m])
            var q_l_off = (e_l * NP + n_l) * NC
            var flux_off = (fid * NFP + m) * NC
            if bc_type != Int32(0):
                _ = physics.boundary_flux(
                    q_ptr + q_l_off,
                    bc_type, nx, ny,
                    fstar_ptr + flux_off,
                )
            else:
                var n_r = Int(
                    mesh.face_elem_node[(fid * 2 + 1) * NFP + m]
                )
                var q_r_off = (e_r * NP + n_r) * NC
                _ = physics.numerical_flux(
                    q_ptr + q_l_off,
                    q_ptr + q_r_off,
                    nx, ny,
                    fstar_ptr + flux_off,
                )

    # 2) Per-element volume + face summations.
    for elem in range(mesh.num_elements):
        var iJ00 = mesh.elem_invJ[elem * 4 + 0]
        var iJ01 = mesh.elem_invJ[elem * 4 + 1]
        var iJ10 = mesh.elem_invJ[elem * 4 + 2]
        var iJ11 = mesh.elem_invJ[elem * 4 + 3]
        var inv_2A = mesh.elem_inv_2A[elem]

        # Pre-compute internal_flux at every node of this element.
        var elem_flux_buf = List[Float64]()
        for _ in range(NP * 2 * NC):
            elem_flux_buf.append(0.0)
        var elem_flux_ptr = rebind[UnsafePointer[Float64, MutAnyOrigin]](
            elem_flux_buf.unsafe_ptr()
        )
        for j in range(NP):
            _ = physics.internal_flux(
                q_ptr + (elem * NP + j) * NC,
                elem_flux_ptr + j * 2 * NC,
            )

        for i in range(NP):
            var x_i = mesh.elem_node_xyz[(elem * NP + i) * 2 + 0]
            var y_i = mesh.elem_node_xyz[(elem * NP + i) * 2 + 1]

            # Source at node i (many physics are no-op here).
            var source_buf = List[Float64]()
            for _ in range(NC):
                source_buf.append(0.0)
            var source_ptr = rebind[UnsafePointer[Float64, MutAnyOrigin]](
                source_buf.unsafe_ptr()
            )
            physics.source_term(
                q_ptr + (elem * NP + i) * NC,
                x_i, y_i, source_ptr,
            )

            for c in range(NC):
                # Volume: sum_j D_r[i, j] * (invJ @ F(q_j))_k
                var vol_c: Float64 = 0.0
                for j in range(NP):
                    var fx = elem_flux_buf[j * 2 * NC + 0 * NC + c]
                    var fy = elem_flux_buf[j * 2 * NC + 1 * NC + c]
                    var fr0 = iJ00 * fx + iJ01 * fy
                    var fr1 = iJ10 * fx + iJ11 * fy
                    var D_r = re.D_ref[0 * NP * NP + i * NP + j]
                    var D_s = re.D_ref[1 * NP * NP + i * NP + j]
                    vol_c += fr0 * D_r + fr1 * D_s

                # Face: sum over 3 edges of sign * face_len * Lift * fstar
                var face_c: Float64 = 0.0
                for lf in range(3):
                    var side = Int(mesh.elem_face_side[elem * 3 + lf])
                    var sign: Float64
                    if side == 0:
                        sign = 1.0
                    else:
                        sign = -1.0
                    var fid = Int(mesh.elem_faces[elem * 3 + lf])
                    var face_len = mesh.face_length[fid]
                    for m in range(NFP):
                        var r = Int(
                            mesh.elem_canon_to_ref[(elem * 3 + lf) * NFP + m]
                        )
                        var Lim = re.Lift_ref[lf * NP * NFP + i * NFP + r]
                        var fstar_c = fstar_buf[(fid * NFP + m) * NC + c]
                        face_c += sign * face_len * Lim * fstar_c

                rhs[(elem * NP + i) * NC + c] = (
                    vol_c - inv_2A * face_c + source_buf[c]
                )


# ----------------------------------------------------------------------
# SSPRK3 time step wrapper (physics-generic)
# ----------------------------------------------------------------------
# Standard Gottlieb-Shu SSPRK3:
#   q1   = q       + dt * L(q)
#   q2   = 3/4 * q + 1/4 * (q1 + dt * L(q1))
#   qnew = 1/3 * q + 2/3 * (q2 + dt * L(q2))
# where L is `dg_rhs_2d[P, PhysT]`.  Mutates `q` in place on the final
# stage.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Barth-Jespersen slope limiter (2D, Venkatakrishnan-smoothed)
# ----------------------------------------------------------------------
# Conservation-preserving post-stage limiter.  For each element, scales
# every nodal deviation from the element's mean by the tightest theta
# that keeps the scaled deviation within the (min, max) cell-average
# range of the element + its 3 face neighbours (sampled on component 0).
# Venkat smoothing avoids over-limiting smooth flows (epsilon=0 recovers
# raw BJ, which kills high-order accuracy even in smooth regions).
#
# Apply theta uniformly across all NC components so physically coupled
# quantities stay consistent (e.g. momentum and density scale together).
# No-op per-element if theta >= 1 (smooth region).
# ----------------------------------------------------------------------

def bj_limit_2d[P: Int, PhysT: Physics2D](
    mesh: LocalMesh2D[P],
    mut q: List[Float64],
    venkat_eps: Float64 = 0.1,
) raises:
    comptime NC = PhysT.NUM_COMPONENTS
    comptime NP = num_tri_nodes_2d(P)
    var inv_np = 1.0 / Float64(NP)
    var eps2 = venkat_eps * venkat_eps

    # Pre-compute per-element cell averages of component 0 (density).
    var cell_avg = List[Float64]()
    for _ in range(mesh.num_elements):
        cell_avg.append(0.0)
    for elem in range(mesh.num_elements):
        var s: Float64 = 0.0
        for nn in range(NP):
            s += q[(elem * NP + nn) * NC + 0]
        cell_avg[elem] = s * inv_np

    for elem in range(mesh.num_elements):
        var own_avg = cell_avg[elem]
        # Seed min/max with self so the range is never empty.
        var nbr_min = own_avg
        var nbr_max = own_avg
        for lf in range(3):
            var fid = Int(mesh.elem_faces[elem * 3 + lf])
            var e_l = Int(mesh.face_elem[fid * 2 + 0])
            var e_r = Int(mesh.face_elem[fid * 2 + 1])
            var n = e_r if e_l == elem else e_l
            var a = cell_avg[n]
            if a < nbr_min: nbr_min = a
            if a > nbr_max: nbr_max = a

        # Venkat-smoothed theta on the density component alone.
        var theta: Float64 = 1.0
        var tiny: Float64 = 1.0e-30
        for nn in range(NP):
            var node_val = q[(elem * NP + nn) * NC + 0]
            var delta = node_val - own_avg
            var d_abs = delta if delta >= 0.0 else -delta
            if d_abs <= tiny:
                continue
            var D: Float64
            if delta > 0.0:
                D = nbr_max - own_avg
            else:
                D = own_avg - nbr_min
            if D < 0.0:
                D = 0.0
            var D2 = D * D
            var d2 = d_abs * d_abs
            var Dd = D * d_abs
            var numer = D2 + 2.0 * Dd + eps2
            var denom = D2 + 2.0 * d2 + Dd + eps2
            var alpha = numer / denom
            if alpha < theta:
                theta = alpha

        if not (theta < 1.0):
            continue    # smooth region; leave alone

        # Precompute per-component nodal means *before* we start
        # modifying q (a previous version recomputed these inside the
        # node loop, which read already-scaled values for c>0 and
        # leaked a few percent of mass into the mean).
        var base_vec = InlineArray[Float64, NC](fill=0.0)
        base_vec[0] = own_avg
        for c in range(1, NC):
            var s: Float64 = 0.0
            for m in range(NP):
                s += q[(elem * NP + m) * NC + c]
            base_vec[c] = s * inv_np

        # Apply uniform theta to every node, every component.  Nodal
        # mean is preserved (for a P>1 Lagrange basis the cell average
        # is a weighted sum -- BJ-on-means is the standard limiter
        # convention even though it isn't strictly cell-mean-preserving
        # at high P).
        for nn in range(NP):
            for c in range(NC):
                var offset = (elem * NP + nn) * NC + c
                q[offset] = base_vec[c] + theta * (q[offset] - base_vec[c])


def ssprk2_step_2d[P: Int, PhysT: Physics2D](
    mesh: LocalMesh2D[P],
    re: ReferenceElement2D[P],
    physics: PhysT,
    dt: Float64,
    mut q: List[Float64],
    mut scratch_q1: List[Float64],
    mut scratch_rhs: List[Float64],
) raises:
    """Heun / 2nd-order SSP Runge-Kutta.  Same SSP property as SSPRK3
    but cheaper (2 rhs calls per step vs 3).  Lower CFL limit, so the
    trade-off is problem-dependent; useful for smooth problems where
    the 3rd-order accuracy of SSPRK3 isn't needed."""
    var n = len(q)
    if len(scratch_q1) != n or len(scratch_rhs) != n:
        raise Error("ssprk2_step_2d: scratch buffer size mismatch")

    #  q1 = q + dt L(q)
    dg_rhs_2d[P, PhysT](mesh, re, physics, q, scratch_rhs)
    for k in range(n):
        scratch_q1[k] = q[k] + dt * scratch_rhs[k]

    # q <- 1/2 q + 1/2 (q1 + dt L(q1))
    dg_rhs_2d[P, PhysT](mesh, re, physics, scratch_q1, scratch_rhs)
    for k in range(n):
        q[k] = 0.5 * q[k] + 0.5 * (scratch_q1[k] + dt * scratch_rhs[k])


def rk4_step_2d[P: Int, PhysT: Physics2D](
    mesh: LocalMesh2D[P],
    re: ReferenceElement2D[P],
    physics: PhysT,
    dt: Float64,
    mut q: List[Float64],
    mut scratch_k: List[Float64],
    mut scratch_accum: List[Float64],
    mut scratch_temp: List[Float64],
    mut scratch_rhs: List[Float64],
) raises:
    """Classical RK4: 4 rhs evaluations per step, 4th-order accurate
    for smooth solutions.  Not SSP (no built-in monotonicity), so
    combine with limiters at your own risk on discontinuous problems.
    The four scratch buffers split as:
      * `scratch_k`     -- holds the current stage's rhs
      * `scratch_accum` -- accumulates (k1 + 2 k2 + 2 k3 + k4)
      * `scratch_temp`  -- q + alpha * dt * k_prev for the next stage
      * `scratch_rhs`   -- dg_rhs_2d output slot (alias of scratch_k)"""
    var n = len(q)
    if (len(scratch_k) != n or len(scratch_accum) != n
            or len(scratch_temp) != n or len(scratch_rhs) != n):
        raise Error("rk4_step_2d: scratch buffer size mismatch")
    var half_dt = 0.5 * dt

    # k1 = L(q)
    dg_rhs_2d[P, PhysT](mesh, re, physics, q, scratch_k)
    for k in range(n):
        scratch_accum[k] = scratch_k[k]
        scratch_temp[k] = q[k] + half_dt * scratch_k[k]

    # k2 = L(q + dt/2 k1)
    dg_rhs_2d[P, PhysT](mesh, re, physics, scratch_temp, scratch_k)
    for k in range(n):
        scratch_accum[k] += 2.0 * scratch_k[k]
        scratch_temp[k] = q[k] + half_dt * scratch_k[k]

    # k3 = L(q + dt/2 k2)
    dg_rhs_2d[P, PhysT](mesh, re, physics, scratch_temp, scratch_k)
    for k in range(n):
        scratch_accum[k] += 2.0 * scratch_k[k]
        scratch_temp[k] = q[k] + dt * scratch_k[k]

    # k4 = L(q + dt k3)
    dg_rhs_2d[P, PhysT](mesh, re, physics, scratch_temp, scratch_k)
    for k in range(n):
        scratch_accum[k] += scratch_k[k]

    # q <- q + dt/6 * (k1 + 2 k2 + 2 k3 + k4)
    var one_sixth_dt = dt / 6.0
    for k in range(n):
        q[k] = q[k] + one_sixth_dt * scratch_accum[k]


def ssprk3_step_2d[P: Int, PhysT: Physics2D](
    mesh: LocalMesh2D[P],
    re: ReferenceElement2D[P],
    physics: PhysT,
    dt: Float64,
    mut q: List[Float64],
    mut scratch_q1: List[Float64],
    mut scratch_q2: List[Float64],
    mut scratch_rhs: List[Float64],
    use_limiter: Bool = False,
    venkat_eps: Float64 = 0.1,
) raises:
    var n = len(q)
    if len(scratch_q1) != n or len(scratch_q2) != n or len(scratch_rhs) != n:
        raise Error("ssprk3_step_2d: scratch buffer size mismatch")

    dg_rhs_2d[P, PhysT](mesh, re, physics, q, scratch_rhs)
    for k in range(n):
        scratch_q1[k] = q[k] + dt * scratch_rhs[k]
    if use_limiter:
        bj_limit_2d[P, PhysT](mesh, scratch_q1, venkat_eps)

    dg_rhs_2d[P, PhysT](mesh, re, physics, scratch_q1, scratch_rhs)
    for k in range(n):
        scratch_q2[k] = (
            0.75 * q[k]
            + 0.25 * (scratch_q1[k] + dt * scratch_rhs[k])
        )
    if use_limiter:
        bj_limit_2d[P, PhysT](mesh, scratch_q2, venkat_eps)

    dg_rhs_2d[P, PhysT](mesh, re, physics, scratch_q2, scratch_rhs)
    for k in range(n):
        q[k] = (
            (1.0 / 3.0) * q[k]
            + (2.0 / 3.0) * (scratch_q2[k] + dt * scratch_rhs[k])
        )
    if use_limiter:
        bj_limit_2d[P, PhysT](mesh, q, venkat_eps)


# ----------------------------------------------------------------------
# Back-compat wrappers -- existing tests / drivers still call these names
# ----------------------------------------------------------------------

def advection_rhs_2d[P: Int](
    mesh: LocalMesh2D[P],
    re: ReferenceElement2D[P],
    vx: Float64, vy: Float64,
    q_in: List[Float64],
    mut rhs: List[Float64],
) raises:
    var physics = Advection2D(vx, vy)
    dg_rhs_2d[P, Advection2D](mesh, re, physics, q_in, rhs)
