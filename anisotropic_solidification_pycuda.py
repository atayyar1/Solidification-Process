import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule

# Grid size
nx = ny = 128
hx = hy = 1.0 / nx
ht = 1e-3
steps = 10

# Parameters from anisotropic model
DT = 1e-4      # thermal diffusivity
TAU = 1.0      # relaxation time
k1 = 0.5
k2 = 1.0
alpha = 1.0
c_aniso = 0.05
mode = 4
angle0 = 0.0

# Allocate fields
phi_host = np.zeros((ny, nx), dtype=np.float64)
temp_host = np.zeros((ny, nx), dtype=np.float64)

# Initial seed
r0 = 0.1
for j in range(ny):
    for i in range(nx):
        x = (i-nx/2)*hx
        y = (j-ny/2)*hy
        r = np.sqrt(x*x+y*y)
        phi_host[j,i] = 0.5*(1-np.tanh((r-r0)/0.01))

# Setup GPU
drv.init()
ctx = drv.Device(0).make_context()
cc = ''.join(str(x) for x in ctx.get_device().compute_capability())

phi = drv.mem_alloc(phi_host.nbytes)
phi_new = drv.mem_alloc(phi_host.nbytes)
temp = drv.mem_alloc(temp_host.nbytes)
temp_new = drv.mem_alloc(temp_host.nbytes)
drv.memcpy_htod(phi, phi_host)
drv.memcpy_htod(temp, temp_host)

kernel_src = f"""
#include <math.h>
#define NX {nx}
#define NY {ny}
#define HX {hx}
#define HY {hy}
#define HT {ht}
#define DT {DT}
#define TAU {TAU}
#define K1 {k1}
#define K2 {k2}
#define ALPHA {alpha}
#define CANISO {c_aniso}
#define MODE {mode}
#define ANG0 {angle0}

__global__ void step(double* phi, double* temp, double* phi_new, double* temp_new){{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    if(i>=NX || j>=NY) return;

    int ip = min(i+1,NX-1);
    int im = max(i-1,0);
    int jp = min(j+1,NY-1);
    int jm = max(j-1,0);

    double phix = (phi[j*NX+ip]-phi[j*NX+im])/(2*HX);
    double phiy = (phi[jp*NX+i]-phi[jm*NX+i])/(2*HY);
    double angle = atan2(phiy, phix);
    double beta = cos(MODE*(angle-ANG0));
    double dbeta = -MODE*sin(MODE*(angle-ANG0));

    double a = 1.0 + CANISO*beta;
    double dxx = ALPHA*a*a;
    double dxy = -ALPHA*a*CANISO*dbeta;
    double dyy = dxx;

    double phixx = (phi[j*NX+ip]-2*phi[j*NX+i]+phi[j*NX+im])/(HX*HX);
    double phiyy = (phi[jp*NX+i]-2*phi[j*NX+i]+phi[jm*NX+i])/(HY*HY);
    double phixy = (phi[jp*NX+ip]-phi[jp*NX+im]-phi[jm*NX+ip]+phi[jm*NX+im])/(4*HX*HY);
    double div = dxx*phixx + dyy*phiyy + 2.0*dxy*phixy;

    double rhs_phi = div + (phi[j*NX+i] - 0.5) - (K1/M_PI)*atan(K2*temp[j*NX+i])*(1.0-phi[j*NX+i]);
    double dphi = rhs_phi/TAU;
    phi_new[j*NX+i] = phi[j*NX+i] + HT*dphi;

    double lapT = (temp[j*NX+ip]+temp[j*NX+im]+temp[jp*NX+i]+temp[jm*NX+i]-4*temp[j*NX+i])/(HX*HX);
    temp_new[j*NX+i] = temp[j*NX+i] + HT*(DT*lapT + dphi);
}}
"""

mod = SourceModule(kernel_src, arch='sm_'+cc)
step = mod.get_function('step')
threads=(16,16,1)
blocks=((nx+15)//16, (ny+15)//16,1)

for _ in range(steps):
    step(phi, temp, phi_new, temp_new, block=threads, grid=blocks)
    step(phi_new, temp_new, phi, temp, block=threads, grid=blocks)

result = np.empty_like(phi_host)
drv.memcpy_dtoh(result, phi)
print('mean phi:', result.mean())
ctx.pop()
