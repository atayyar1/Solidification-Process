import numpy as np
import pycuda.autoinit  # initializes CUDA driver
from pycuda import driver as drv
from pycuda.compiler import SourceModule

# -------------------------------
# Numerical parameters
# -------------------------------
dx = dy = 0.025
nx = ny = 500

dt = 5e-4
steps = 10000

tau = 3e-4
kappa = 2.25

zeta = 0.02
aniso = 6.0
angle0 = np.pi / 8

kappa1 = 0.9
kappa2 = 20.0

r0 = 5 * dx
width = dx

# -------------------------------
# Allocate fields on host
# -------------------------------
shape = (ny, nx)
phi_host = np.zeros(shape, dtype=np.float64)
phi_new_host = np.zeros_like(phi_host)

temp_host = np.full(shape, -2.25, dtype=np.float64)
# Buffer for updated temperature
temp_new_host = np.zeros_like(temp_host)

# gradient and laplacian storage
grad_phix_host = np.zeros_like(phi_host)
grad_phiy_host = np.zeros_like(phi_host)
lap_phi_host = np.zeros_like(phi_host)
lap_temp_host = np.zeros_like(phi_host)
a2_host = np.zeros_like(phi_host)
ax_host = np.zeros_like(phi_host)
ay_host = np.zeros_like(phi_host)

# -------------------------------
# Initial condition: circular seed
# -------------------------------
for j in range(ny):
    y = (j - ny/2) * dy
    for i in range(nx):
        x = (i - nx/2) * dx
        r = np.hypot(x, y)
        phi_host[j, i] = 0.5 * (1.0 - np.tanh((r - r0) / width))

# -------------------------------
# CUDA kernels
# -------------------------------
block_size_string = "#define block_size_x 16\n#define block_size_y 16\n"

kernel_calcgrad = f"""
#include <math.h>
#define nx {nx}
#define ny {ny}
#define dx {dx}
#define dy {dy}
#define pi {np.pi}
#define zeta {zeta}
#define aniso {aniso}
#define angle0 {angle0}

__global__ void calcgrad(
    const double *phi,
    const double *temp,
    double *grad_phix,
    double *grad_phiy,
    double *lap_phi,
    double *lap_temp,
    double *ax,
    double *ay,
    double *a2)
{{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int ip = (i + 1) % nx, im = (i - 1 + nx) % nx;
    int jp = (j + 1) % ny, jm = (j - 1 + ny) % ny;
    int idx    = j*nx + i;
    int idx_ip = j*nx + ip, idx_im = j*nx + im;
    int idx_jp = jp*nx + i, idx_jm = jm*nx + i;

    // gradients
    grad_phix[idx] = (phi[idx_ip] - phi[idx_im]) / (2.0 * dx);
    grad_phiy[idx] = (phi[idx_jp] - phi[idx_jm]) / (2.0 * dy);

    // 9-point laplacian (dimensionless)
    lap_phi[idx] = (
        2.0*(phi[idx_ip] + phi[idx_im] + phi[idx_jp] + phi[idx_jm]) +
        phi[jp*nx+ip] + phi[jm*nx+im] + phi[jp*nx+im] + phi[jm*nx+ip] -
        12.0*phi[idx]
    ) / (3.0 * dx * dx);

    lap_temp[idx] = (
        2.0*(temp[idx_ip] + temp[idx_im] + temp[idx_jp] + temp[idx_jm]) +
        temp[jp*nx+ip] + temp[jm*nx+im] + temp[jp*nx+im] + temp[jm*nx+ip] -
        12.0*temp[idx]
    ) / (3.0 * dx * dx);

    // anisotropy angle
    double gx = grad_phix[idx];
    double gy = grad_phiy[idx];
    double ang;
    if (gx == 0.0)
        ang = (gy > 0.0) ? 0.5*pi : -0.5*pi;
    else if (gx > 0.0)
        ang = (gy >= 0.0) ? atan(gy/gx) : 2.0*pi + atan(gy/gx);
    else
        ang = pi + atan(gy/gx);

    double a  = 1.0 + zeta * cos(aniso*(ang - angle0));
    double da = -aniso*zeta * sin(aniso*(ang - angle0));

    ax[idx] =  a * da * gx;
    ay[idx] = -a * da * gy;
    a2[idx] =  a * a;
}}
"""

from string import Template
kernel_timeevol = Template(r"""
#include <math.h>
#define nx     $nx
#define ny     $ny
#define dx     $dx
#define dy     $dy
#define dt     $dt
#define tau    $tau
#define kappa  $kappa
#define kappa1 $kappa1
#define kappa2 $kappa2
#define pi     3.141592653589793

__global__ void timeevol(
    const double *phi,
    const double *temp,
    double *phi_new,
    double *temp_new,
    const double *ax,
    const double *ay,
    const double *a2,
    const double *grad_phix,
    const double *grad_phiy,
    const double *lap_phi,
    const double *lap_temp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int ip = (i + 1) % nx, im = (i - 1 + nx) % nx;
    int jp = (j + 1) % ny, jm = (j - 1 + ny) % ny;

    int idx    = j*nx + i;
    int idx_ip = j*nx + ip, idx_im = j*nx + im;
    int idx_jp = jp*nx + i, idx_jm = jm*nx + i;

    double d_ay_dx = (ay[idx_ip] - ay[idx_im]) / (2.0 * dx);
    double d_ax_dy = (ax[idx_jp] - ax[idx_jm]) / (2.0 * dy);
    double d_a2_dx = (a2[idx_ip] - a2[idx_im]) / (2.0 * dx);
    double d_a2_dy = (a2[idx_jp] - a2[idx_jm]) / (2.0 * dy);
    double div_flux = d_ay_dx + d_ax_dy
                      + a2[idx] * lap_phi[idx]
                      + d_a2_dx * grad_phix[idx]
                      + d_a2_dy * grad_phiy[idx];

    double xi     = phi[idx];
    double deltaT = -temp[idx];
    double m      = (xi - 0.5) - (kappa1/pi) * atan(kappa2 * deltaT);
    double source = xi * (1.0 - xi) * m;
    double dxi_dt = (div_flux + source) / tau;
    double phi_tmp = xi + dt * dxi_dt;
    if (phi_tmp < 0.0) phi_tmp = 0.0;
    else if (phi_tmp > 1.0) phi_tmp = 1.0;
    phi_new[idx] = phi_tmp;

    double dT_dt = kappa * lap_temp[idx] + dxi_dt;
    double temp_tmp = temp[idx] + dt * dT_dt;
    if (temp_tmp < -1.0) temp_tmp = -1.0;
    else if (temp_tmp > 0.0) temp_tmp = 0.0;
    temp_new[idx] = temp_tmp;
}
""").substitute(
    nx=nx, ny=ny,
    dx=f"{dx:.6e}", dy=f"{dy:.6e}",
    dt=f"{dt:.6e}", tau=f"{tau:.6e}",
    kappa=f"{kappa:.6e}",
    kappa1=f"{kappa1:.6e}",
    kappa2=f"{kappa2:.6e}"
)

# -------------------------------
# Compile kernels
# -------------------------------
mod_grad = SourceModule(block_size_string + kernel_calcgrad)
calcgrad = mod_grad.get_function("calcgrad")

mod_evol = SourceModule(block_size_string + kernel_timeevol)
timeevol = mod_evol.get_function("timeevol")

threads = (16, 16, 1)
grid = (nx // threads[0], ny // threads[1], 1)

# -------------------------------
# Allocate device memory
# -------------------------------
phi      = drv.mem_alloc(phi_host.nbytes)
phi_new  = drv.mem_alloc(phi_host.nbytes)

temp     = drv.mem_alloc(temp_host.nbytes)
temp_new = drv.mem_alloc(temp_host.nbytes)

grad_phix = drv.mem_alloc(phi_host.nbytes)
grad_phiy = drv.mem_alloc(phi_host.nbytes)
ax        = drv.mem_alloc(phi_host.nbytes)
ay        = drv.mem_alloc(phi_host.nbytes)
a2        = drv.mem_alloc(phi_host.nbytes)
lap_phi   = drv.mem_alloc(phi_host.nbytes)
lap_temp  = drv.mem_alloc(phi_host.nbytes)

# -------------------------------
# Copy initial data to GPU
# -------------------------------
drv.memcpy_htod(phi, phi_host)
drv.memcpy_htod(temp, temp_host)

# -------------------------------
# Main time loop
# -------------------------------
for step in range(steps):
    calcgrad(
        phi, temp, grad_phix, grad_phiy, lap_phi, lap_temp,
        ax, ay, a2,
        block=threads, grid=grid)

    timeevol(
        phi, temp, phi_new, temp_new,
        ax, ay, a2,
        grad_phix, grad_phiy, lap_phi, lap_temp,
        block=threads, grid=grid)

    phi, phi_new = phi_new, phi
    temp, temp_new = temp_new, temp

# Retrieve results
phi_result = np.empty_like(phi_host)
temp_result = np.empty_like(temp_host)
drv.memcpy_dtoh(phi_result, phi)
drv.memcpy_dtoh(temp_result, temp)

print("phi min/max:", phi_result.min(), phi_result.max())
print("temp min/max:", temp_result.min(), temp_result.max())
