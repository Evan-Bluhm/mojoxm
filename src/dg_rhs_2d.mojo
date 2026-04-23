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
from src.boundary import BC_INTERIOR, BC_WALL, BC_OUTFLOW


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

@fieldwise_init
struct Advection2D(Physics2D, ImplicitlyCopyable, Movable):
    comptime NUM_COMPONENTS = 1

    var vx: Float64
    var vy: Float64

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
        if bc_type == BC_OUTFLOW:
            # Zero-gradient: ghost = interior.  Upwind becomes
            # interior-side on outflow, zero on inflow.
            flux[0] = vn * q_int[0]
        else:
            # BC_WALL and anything else: zero-Dirichlet ghost.
            if vn >= 0.0:
                flux[0] = vn * q_int[0]
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

@fieldwise_init
struct Euler2D(Physics2D, ImplicitlyCopyable, Movable):
    comptime NUM_COMPONENTS = 4

    var gamma: Float64
    var min_density: Float64
    var min_pressure: Float64

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
            # In (x, y) coords with normal (nx, ny):
            #   m_n = mx * nx + my * ny
            #   m_ghost = m - 2 m_n * (nx, ny)
            var mx = q_int[1]
            var my = q_int[2]
            var m_n = mx * nx + my * ny
            q_g_buf[0] = q_int[0]
            q_g_buf[1] = mx - 2.0 * m_n * nx
            q_g_buf[2] = my - 2.0 * m_n * ny
            q_g_buf[3] = q_int[3]
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

@fieldwise_init
struct ShallowWater2D(Physics2D, ImplicitlyCopyable, Movable):
    comptime NUM_COMPONENTS = 3

    var g: Float64            # gravitational acceleration
    var min_h: Float64        # depth floor (avoids divide-by-zero at dry patches)

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
        for m in range(NFP):
            var n_l = Int(mesh.face_elem_node[(fid * 2 + 0) * NFP + m])
            var n_r = Int(mesh.face_elem_node[(fid * 2 + 1) * NFP + m])
            var q_l_off = (e_l * NP + n_l) * NC
            var q_r_off = (e_r * NP + n_r) * NC
            var flux_off = (fid * NFP + m) * NC
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

def ssprk3_step_2d[P: Int, PhysT: Physics2D](
    mesh: LocalMesh2D[P],
    re: ReferenceElement2D[P],
    physics: PhysT,
    dt: Float64,
    mut q: List[Float64],
    mut scratch_q1: List[Float64],
    mut scratch_q2: List[Float64],
    mut scratch_rhs: List[Float64],
) raises:
    var n = len(q)
    if len(scratch_q1) != n or len(scratch_q2) != n or len(scratch_rhs) != n:
        raise Error("ssprk3_step_2d: scratch buffer size mismatch")

    dg_rhs_2d[P, PhysT](mesh, re, physics, q, scratch_rhs)
    for k in range(n):
        scratch_q1[k] = q[k] + dt * scratch_rhs[k]

    dg_rhs_2d[P, PhysT](mesh, re, physics, scratch_q1, scratch_rhs)
    for k in range(n):
        scratch_q2[k] = (
            0.75 * q[k]
            + 0.25 * (scratch_q1[k] + dt * scratch_rhs[k])
        )

    dg_rhs_2d[P, PhysT](mesh, re, physics, scratch_q2, scratch_rhs)
    for k in range(n):
        q[k] = (
            (1.0 / 3.0) * q[k]
            + (2.0 / 3.0) * (scratch_q2[k] + dt * scratch_rhs[k])
        )


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
