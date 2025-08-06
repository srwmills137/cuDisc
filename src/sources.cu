#include <iostream>
#include <cuda_runtime.h>

#include "grid.h"
#include "field.h"
#include "cuda_array.h"
#include "dustdynamics.h"
#include "gasdynamics.h"
#include "constants.h"
#include "sources.h"
#include "drag_const.h"

#include "coagulation/size_grid.h"

// Simple Scheme

// Compute Van Leer limited slope for derivative functions
__device__
double _vl_slope_gas(double dQF, double dQB, double cF, double cB) {

    if (dQF*dQB > 0.) {
        double v = dQB/dQF ;
        return dQB * (cF*v + cB) / (v*v + (cF + cB - 2)*v + 1.) ;
    } 
    else {
        return 0. ;
    }
}

// Compute r derivative
__device__
double vl_r2D(GridRef& g, FieldRef<double>& Qty, int i, int j) {

    double rc = (g.rc(i,j));

    double cF = ((g.rc(i+1,j)) - rc) / ((g.re(i+1,j))-rc) ;
    double cB = ((g.rc(i-1,j)) - rc) / ((g.re(i,j))-rc) ;

    double dQF = (Qty(i+1, j) - Qty(i, j)) / ((g.rc(i+1,j)) - rc) ;
    double dQB = (Qty(i-1, j) - Qty(i, j)) / ((g.rc(i-1,j)) - rc) ;

    return _vl_slope_gas(dQF, dQB, cF, cB) ;
}

// compute Z derivative
__device__
double vl_Z2D(GridRef& g, FieldRef<double>& Qty, int i, int j) {

    double Zc = g.Zc(i,j);

    double cF = (g.Zc(i,j+1) - Zc) / (g.Ze(i,j+1)-Zc) ;
    double cB = (g.Zc(i,j-1) - Zc) / (g.Ze(i,j)-Zc) ;

    double dQF = (Qty(i, j+1) - Qty(i, j)) / (g.Zc(i,j+1) - Zc) ;
    double dQB = (Qty(i, j-1) - Qty(i, j)) / (g.Zc(i,j-1) - Zc) ;

    return _vl_slope_gas(dQF, dQB, cF, cB) ;
}

// compute R derivative
__device__
double vl_R2D(GridRef& g, FieldRef<double>& Qty, int i, int j) {
    return (vl_r2D(g, Qty, i, j) - g.sin_th_c(j) * vl_Z2D(g, Qty, i, j)) / g.cos_th_c(j) ;
}

// compute TRphi and TZphi components of the stress tensor field
__global__
void _calc_T(GridRef g, FieldRef<Prims> wg, FieldRef<double> TRphi, FieldRef<double> TZphi, FieldRef<double> vphi, const double* nu) {
    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            TRphi(i,j) = wg(i,j).rho * nu[i] * (vl_R2D(g, vphi, i, j) - vphi(i,j) / g.Rc(i)) ;
            TZphi(i,j) = wg(i,j).rho * nu[i] * vl_Z2D(g, vphi, i, j) ;
        }
    }
}

// calculate the pressure field
__global__
void _calc_pressure(GridRef g, FieldRef<Prims> wg, FieldConstRef<double> cs, FieldRef<double> p) {
    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            p(i,j) = wg(i, j).rho * cs(i, j) * cs(i, j) ;
        }
    }
}

// upload the phi component of velocity into a Field
__global__
void _get_vphi(GridRef g, FieldRef<Prims> wg, FieldRef<double> vphi) {
    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            vphi(i,j) = wg(i, j).v_phi ;
        }
    }
}

// Computes the Keplerian angluar velocity squared
__device__
double OmK2(GridRef& g, double Mstar, int i, int j) {

    return GMsun * Mstar / std::pow(g.Rc(i)*g.Rc(i)+g.Zc(i,j)*g.Zc(i,j), 1.5);

}

// Computes the source terms from curvature and gravtiy (for dust)
__global__
void _source_curv_grav(GridRef g, Field3DRef<Prims> w, Field3DRef<Quants> u, FieldConstRef<Prims> wg, double dt, double Mstar, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            for (int k=kidx; k<w.Nd; k+=kstride) {

                if (w(i,j,k).rho > 1.1*wg(i,j).rho*floor) {

                    double f1 = w(i,j,k).v_phi*w(i,j,k).v_phi/g.Rc(i) - OmK2(g, Mstar, i, j)*g.Rc(i);
                    double f2 = -OmK2(g, Mstar, i, j)*g.Zc(i,j);

                    u(i,j,k).mom_R += dt * f1 * w(i,j,k).rho;
                    u(i,j,k).mom_Z += dt * f2 * w(i,j,k).rho;
                }
            }
        }
    }
}

// Computes the source terms from curvature, gravity, and radiative pressure (for dust)
__global__
void _source_curv_grav_pressure(GridRef g, Field3DRef<Prims> w, Field3DRef<Quants> u, FieldConstRef<Prims> wg, Field3DConstRef<double> f_rad, double dt, double Mstar, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            for (int k=kidx; k<w.Nd; k+=kstride) {

                if (w(i,j,k).rho > 1.1*wg(i,j).rho*floor) {

                    double f1 = w(i,j,k).v_phi*w(i,j,k).v_phi/g.Rc(i) - OmK2(g, Mstar, i, j)*g.Rc(i) + f_rad(i,j,k)*g.Rc(i)/(w(i,j,k).rho*g.rc(i,j));
                    double f2 = -OmK2(g, Mstar, i, j)*g.Zc(i,j) + f_rad(i,j,k)*g.Zc(i,j)/(w(i,j,k).rho*g.rc(i,j));

                    u(i,j,k).mom_R += dt * f1 * w(i,j,k).rho;
                    u(i,j,k).mom_Z += dt * f2 * w(i,j,k).rho;

                }
            }
        }
    }
}

// Computes the source terms from curvature, gravity, pressure, and viscosity (for gas)
__global__
void _source_curv_grav_pres_visc(GridRef g, FieldRef<Quants> u, FieldRef<Prims> wg, double dt, double Mstar, FieldRef<double> p,  FieldRef<double> TRphi, FieldRef<double> TZphi, FieldRef<double> vphi, double floor) {
    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {

            double f1 = -vl_R2D(g, p, i, j) + wg(i,j).rho*wg(i,j).v_phi*wg(i,j).v_phi/g.Rc(i) - wg(i,j).rho*OmK2(g, Mstar, i, j)*g.Rc(i) ;
            double f2 = 2*TRphi(i,j) + g.Rc(i)*vl_R2D(g, TRphi, i, j) + g.Rc(i)*vl_Z2D(g, TZphi, i, j) ;
            double f3 = -vl_Z2D(g, p, i, j) - wg(i,j).rho*OmK2(g, Mstar, i, j)*g.Zc(i,j) ;

            u(i,j).mom_R += dt * f1 ;
            u(i,j).amom_phi += dt * f2 ;
            u(i,j).mom_Z += dt * f3 ;
        }
    }
}

// Computes the drag force from the gas onto the dust
__global__
void _source_drag(GridRef g, Field3DRef<Prims> w, FieldConstRef<Prims> w_gas, Field3DConstRef<double> t_stop, double dt, double Mstar) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            for (int k=kidx; k<w.Nd; k+=kstride) {

                // semi-implicit eulerian [1-dt*df/dy]*dy = dt*f where y=(momR, vphi, momZ) and f is the vector of drag terms

                double dv_R   = - (w(i,j,k).v_R - w_gas(i,j).v_R);
                double dv_phi = - (w(i,j,k).v_phi - w_gas(i,j).v_phi);
                double dv_Z   = - (w(i,j,k).v_Z - w_gas(i,j).v_Z);

                double ft = dt/(dt + t_stop(i,j,k)) ;

                w(i,j,k).v_R += ft * dv_R ;
                w(i,j,k).v_phi += ft * dv_phi ;
                w(i,j,k).v_Z += ft * dv_Z ;


                double max_v = 5.e5; 

                if (w(i,j,k).v_Z > max_v) { w(i,j,k).v_Z = max_v; }
                if (w(i,j,k).v_Z < -max_v) { w(i,j,k).v_Z = -max_v; }
                if (w(i,j,k).v_R > max_v) { w(i,j,k).v_R = max_v; }
                if (w(i,j,k).v_R < -max_v) { w(i,j,k).v_R = -max_v; }
            }
        }
    }

}

// Computes the stopping time for the dust
template<bool full_stokes>
__global__
void _calc_t_s(GridRef g, Field3DConstRef<Prims> q, FieldConstRef<Prims> w_gas, FieldConstRef<double> T, 
                    Field3DRef<double> t_stop, const RealType* s, double rho_m, double mu) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int kidx = threadIdx.z + blockIdx.z*blockDim.z ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;
    int kstride = gridDim.z * blockDim.z ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            for (int k=kidx; k<q.Nd; k+=kstride) {
                double cs = sqrt(k_B*T(i,j)/(mu*m_H));
                t_stop(i,j,k) = calc_t_s<full_stokes>(q(i,j,k), w_gas(i,j), s[k], rho_m, cs, mu);
            }
        }
    }
}

// Computes explicit source terms for dust
template<bool use_full_stokes>
void Sources<use_full_stokes>::source_exp(Grid& g, Field3D<Prims>& w, Field3D<Quants>& u, double dt) {

    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (u.Nd+7)/8) ;

    _source_curv_grav<<<blocks,threads>>>(g, w, u, _w_gas, dt, _Mstar, _floor);
}

// Computes implicit source terms for dust
template<bool use_full_stokes>
void Sources<use_full_stokes>::source_imp(Grid& g, Field3D<Prims>& w, double dt) {

    Field3D<double> t_stop = Field3D<double>(g.NR+2*g.Nghost,g.Nphi+2*g.Nghost,w.Nd);

    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (w.Nd+7)/8) ;

    if (use_full_stokes) 
        _calc_t_s<true><<<blocks,threads>>>(g, w, _w_gas, _T, t_stop, _sizes.grain_sizes(), _sizes.solid_density(), _mu);
    else
        _calc_t_s<false><<<blocks,threads>>>(g, w, _w_gas, _T, t_stop, _sizes.grain_sizes(), _sizes.solid_density(), _mu);
    _source_drag<<<blocks,threads>>>(g, w, _w_gas, t_stop, dt, _Mstar);
}

// Computes explicit source terms for dust with radiative pressure on
template<bool use_full_stokes>
void SourcesRad<use_full_stokes>::source_exp(Grid& g, Field3D<Prims>& w, Field3D<Quants>& u, double dt) {

    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (u.Nd+7)/8) ;

    _source_curv_grav_pressure<<<blocks,threads>>>(g, w, u, _w_gas, _f_rad, dt, _Mstar, _floor);
}

// Computes implicit source terms for dust with radiative pressure on
template<bool use_full_stokes>
void SourcesRad<use_full_stokes>::source_imp(Grid& g, Field3D<Prims>& w, double dt) {

    Field3D<double> t_stop = Field3D<double>(g.NR+2*g.Nghost,g.Nphi+2*g.Nghost,w.Nd);

    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, (w.Nd+7)/8) ;

    if (use_full_stokes) 
        _calc_t_s<true><<<blocks,threads>>>(g, w, _w_gas, _T, t_stop, _sizes.grain_sizes(), _sizes.solid_density(), _mu);
    else
        _calc_t_s<false><<<blocks,threads>>>(g, w, _w_gas, _T, t_stop, _sizes.grain_sizes(), _sizes.solid_density(), _mu);
    _source_drag<<<blocks,threads>>>(g, w, _w_gas, t_stop, dt, _Mstar);
}

// Computes explicit source terms for gas
template<bool use_full_stokes>
void SourcesGas<use_full_stokes>::source_exp(Grid& g, Field<Prims>& w_g, Field<Quants>& u, const double* nu, FieldConstRef<double> cs, double dt) {
    Field<double> TRphi = create_field<double>(g);
    Field<double> TZphi = create_field<double>(g);
    Field<double> p = create_field<double>(g);
    Field<double> vphi = create_field<double>(g);
    
    dim3 threads(16,8,8);
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, 1) ;

    _get_vphi<<<blocks,threads>>>(g, w_g, vphi) ;
    // find pressure
    _calc_pressure<<<blocks,threads>>>(g, w_g, cs, p) ;
    // calculate the stress tensor components
    _calc_T<<<blocks,threads>>>(g, w_g, TRphi, TZphi, vphi, nu) ;
    // compute the total source terms
    _source_curv_grav_pres_visc<<<blocks,threads>>>(g, u, w_g, dt, _Mstar, p, TRphi, TZphi, vphi, _floor);
}

template class Sources<true>;
template class Sources<false>;
template class SourcesRad<true>;
template class SourcesRad<false>;
template class SourcesGas<true>;
template class SourcesGas<false>;