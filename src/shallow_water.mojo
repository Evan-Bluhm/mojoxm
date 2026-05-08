# ======================================================================
# Shallow-water physics (2D embedded in 3D)
# ======================================================================
#
# The 2D nonlinear shallow-water equations
#
#   dh/dt + d(hu)/dx + d(hv)/dy = 0
#   d(hu)/dt + d(hu^2 + g h^2/2)/dx + d(huv)/dy = 0
#   d(hv)/dt + d(huv)/dx + d(hv^2 + g h^2/2)/dy = 0
#
# embedded in the generic 3D DG solver by treating F^z = 0 everywhere.
# The driver picks a thin 3D mesh (NZ = 2) so the z direction is
# computed but trivial, and the user sees purely 2D wave dynamics.
#
# Conserved state: q = [h, hu, hv].
# Eigenvalues in direction n^ = (nx, ny, nz):
#   u_n - c,  u_n,  u_n + c,      with c = sqrt(g h) and u_n = u nx + v ny
# so the Rusanov dissipation parameter is |u_n| + c.
#
# Bed-slope source (-gh grad b) not implemented in this first cut -- the
# `source_term` hook returns zero, matching a flat-bottomed pool.  A
# driver that wants topography subclasses this struct and overrides
# source_term, or we add it as a follow-up once a real demo needs it.
# ======================================================================

from src.solver import Physics
from src.boundary import BC_WALL, BC_OUTFLOW, BC_INFLOW
from std.math import sqrt


struct ShallowWater(ImplicitlyCopyable, Physics):
    comptime NUM_COMPONENTS = 3

    var g: Float32  # gravitational acceleration
    var h_min: Float32  # depth floor; below this, flux computation
    # treats the cell as dry (c = 0, u = v = 0)
    # BC_INFLOW ghost state (h, h*u, h*v).  Defaults zero.
    var inflow_h: Float32
    var inflow_hu: Float32
    var inflow_hv: Float32

    def __init__(out self, g: Float32, h_min: Float32, inflow_h: Float32 = Float32(0.0), inflow_hu: Float32 = Float32(0.0), inflow_hv: Float32 = Float32(0.0)):
        self.g = g
        self.h_min = h_min
        self.inflow_h = inflow_h
        self.inflow_hu = inflow_hu
        self.inflow_hv = inflow_hv

    # --- DevicePassable plumbing (see std.gpu.host.device_context) ---
    comptime device_type = Self

    def _to_device_type[origin: MutOrigin](self, target: UnsafePointer[NoneType, origin]):
        target.bitcast[Self]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "ShallowWater"

    # Internal flux tensor at a single node.  flux[d*3 + c] = F^d_c.
    # F^x = [hu, hu^2 + g h^2/2, huv]
    # F^y = [hv, huv, hv^2 + g h^2/2]
    # F^z = 0
    def internal_flux(self, q: UnsafePointer[Float32, MutAnyOrigin], flux: UnsafePointer[Float32, MutAnyOrigin]) -> Float32:
        var h = q[0] if q[0] > self.h_min else self.h_min
        var hu = q[1]
        var hv = q[2]
        var u = hu / h
        var v = hv / h
        var gh2_2 = Float32(0.5) * self.g * h * h
        # F^x
        flux[0 * 3 + 0] = hu
        flux[0 * 3 + 1] = hu * u + gh2_2
        flux[0 * 3 + 2] = hu * v
        # F^y
        flux[1 * 3 + 0] = hv
        flux[1 * 3 + 1] = hu * v
        flux[1 * 3 + 2] = hv * v + gh2_2
        # F^z = 0 (purely 2D system)
        flux[2 * 3 + 0] = Float32(0.0)
        flux[2 * 3 + 1] = Float32(0.0)
        flux[2 * 3 + 2] = Float32(0.0)
        # Loose CFL bound; per-face numerical_flux return is tighter.
        var c = sqrt(self.g * h)
        var uu = u if u >= Float32(0.0) else -u
        var vv = v if v >= Float32(0.0) else -v
        return max(uu, vv) + c

    # Rusanov numerical flux at a face with normal (nx, ny, nz).
    # alpha = max over both sides of |u_n| + sqrt(g h).
    def numerical_flux(
        self,
        q_l: UnsafePointer[Float32, MutAnyOrigin],
        q_r: UnsafePointer[Float32, MutAnyOrigin],
        nx: Float32,
        ny: Float32,
        nz: Float32,
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        var hl = q_l[0] if q_l[0] > self.h_min else self.h_min
        var hul = q_l[1]
        var hvl = q_l[2]
        var ul = hul / hl
        var vl = hvl / hl
        var un_l = ul * nx + vl * ny
        var cl = sqrt(self.g * hl)
        var gh2_2_l = Float32(0.5) * self.g * hl * hl

        var hr = q_r[0] if q_r[0] > self.h_min else self.h_min
        var hur = q_r[1]
        var hvr = q_r[2]
        var ur = hur / hr
        var vr = hvr / hr
        var un_r = ur * nx + vr * ny
        var cr = sqrt(self.g * hr)
        var gh2_2_r = Float32(0.5) * self.g * hr * hr

        # F_n(q_l)
        var Fl0 = hul * nx + hvl * ny
        var Fl1 = hul * un_l + gh2_2_l * nx
        var Fl2 = hvl * un_l + gh2_2_l * ny
        # F_n(q_r)
        var Fr0 = hur * nx + hvr * ny
        var Fr1 = hur * un_r + gh2_2_r * nx
        var Fr2 = hvr * un_r + gh2_2_r * ny

        var aun_l = un_l if un_l >= Float32(0.0) else -un_l
        var aun_r = un_r if un_r >= Float32(0.0) else -un_r
        var alpha = max(aun_l + cl, aun_r + cr)

        var half = Float32(0.5)
        flux[0] = half * (Fl0 + Fr0) - half * alpha * (q_r[0] - q_l[0])
        flux[1] = half * (Fl1 + Fr1) - half * alpha * (q_r[1] - q_l[1])
        flux[2] = half * (Fl2 + Fr2) - half * alpha * (q_r[2] - q_l[2])
        return alpha

    # Boundary-face flux via ghost-state synthesis.  BC_WALL reflects
    # normal momentum (slip wall); BC_OUTFLOW is zero-gradient.
    def boundary_flux(self, q_int: UnsafePointer[Float32, MutAnyOrigin], bc_type: Int32, nx: Float32, ny: Float32, nz: Float32, flux: UnsafePointer[Float32, MutAnyOrigin]) -> Float32:
        var q_ghost = InlineArray[Float32, 3](fill=0.0)
        if bc_type == BC_WALL:
            # Reflect the (xy) normal momentum; keep tangential.
            q_ghost[0] = q_int[0]
            var mn = q_int[1] * nx + q_int[2] * ny
            q_ghost[1] = q_int[1] - Float32(2.0) * mn * nx
            q_ghost[2] = q_int[2] - Float32(2.0) * mn * ny
        elif bc_type == BC_INFLOW:
            q_ghost[0] = self.inflow_h
            q_ghost[1] = self.inflow_hu
            q_ghost[2] = self.inflow_hv
        else:
            # BC_OUTFLOW (default): zero-gradient ghost.
            q_ghost[0] = q_int[0]
            q_ghost[1] = q_int[1]
            q_ghost[2] = q_int[2]
        var q_ghost_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](q_ghost.unsafe_ptr())
        return self.numerical_flux(q_int, q_ghost_p, nx, ny, nz, flux)

    # Flat-bottom shallow water: no source.  A bed-slope term
    # -g h grad(b) would fit naturally here once we wire up a bed
    # elevation field; for now this is a no-op.
    def source_term(self, q: UnsafePointer[Float32, MutAnyOrigin], x: Float32, y: Float32, z: Float32, source_out: UnsafePointer[Float32, MutAnyOrigin]):
        source_out[0] = Float32(0.0)
        source_out[1] = Float32(0.0)
        source_out[2] = Float32(0.0)

    def limit_state(self, q: UnsafePointer[Float32, MutAnyOrigin]):
        pass
