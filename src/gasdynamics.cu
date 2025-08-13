#include <iostream>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

#include "advection.h"
#include "diffusion_device.h"
#include "field.h"
#include "grid.h"
#include "reductions.h"
#include "dustdynamics.h"
#include "gasdynamics.h"
#include "scan.h"
#include "constants.h"
#include "utils.h"
#include "sources.h"
#include "van_leer.h"
// Advection solver for gas

// set up the boundary conditions based on the boundary flags
__global__
void _set_boundaries(GridRef g, FieldRef<Prims> w_g, int bound, double floor) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            if (i < g.Nghost) {
                if (bound & BoundaryFlags::open_R_inner) {  //outflow
                    if (w_g(g.Nghost,j).v_R < 0.) {
                        w_g(i,j) = w_g(g.Nghost,j) ;
                    }
                    else {
                        w_g(i,j) = w_g(2*g.Nghost-1-i,j) ;
                        w_g(i,j).v_R *= -1 ;
                    }
                }
                else if (bound & BoundaryFlags::set_ext_R_inner) {} //set externally (e.g. inflow)
                else {  //reflecting
                    w_g(i,j) = w_g(2*g.Nghost-1-i,j) ;
                    w_g(i,j).v_R *= -1 ;
                }
            }

            if (j>=g.Nphi+g.Nghost) {
                if (bound & BoundaryFlags::open_Z_outer) {
                    if (w_g(i,g.Nphi+g.Nghost-1).v_R*g.face_normal_Z(i,g.Nphi+g.Nghost).R + 
                        w_g(i,g.Nphi+g.Nghost-1).v_Z*g.face_normal_Z(i,g.Nphi+g.Nghost).Z > 0.) {
                            
                        w_g(i,j) = w_g(i,g.Nphi+g.Nghost-1) ;
                    }
                    else {
                        w_g(i,j)[0] = w_g(i,2*(g.Nphi+g.Nghost)-1-j)[0];
                        w_g(i,j)[1] = w_g(i,2*(g.Nphi+g.Nghost)-1-j)[1] * (g.cos_th(g.Nphi+g.Nghost)*g.cos_th(g.Nphi+g.Nghost) - g.sin_th(g.Nphi+g.Nghost)*g.sin_th(g.Nphi+g.Nghost)) 
                                        + 2.*w_g(i,2*(g.Nphi+g.Nghost)-1-j)[3]*g.sin_th(g.Nphi+g.Nghost)*g.cos_th(g.Nphi+g.Nghost);
                        w_g(i,j)[2] = w_g(i,2*(g.Nphi+g.Nghost)-1-j)[2];
                        w_g(i,j)[3] = w_g(i,2*(g.Nphi+g.Nghost)-1-j)[3] * (-g.cos_th(g.Nphi+g.Nghost)*g.cos_th(g.Nphi+g.Nghost) + g.sin_th(g.Nphi+g.Nghost)*g.sin_th(g.Nphi+g.Nghost))
                                    + 2.*w_g(i,2*(g.Nphi+g.Nghost)-1-j)[1]*g.sin_th(g.Nphi+g.Nghost)*g.cos_th(g.Nphi+g.Nghost);                           
                    }
                }
                else if (bound & BoundaryFlags::set_ext_Z_outer) {} //set externally (e.g. inflow)
                else {
                    w_g(i,j)[0] = w_g(i,2*(g.Nphi+g.Nghost)-1-j)[0];
                    w_g(i,j)[1] = w_g(i,2*(g.Nphi+g.Nghost)-1-j)[1] * (g.cos_th(g.Nphi+g.Nghost)*g.cos_th(g.Nphi+g.Nghost) - g.sin_th(g.Nphi+g.Nghost)*g.sin_th(g.Nphi+g.Nghost)) 
                                    + 2.*w_g(i,2*(g.Nphi+g.Nghost)-1-j)[3]*g.sin_th(g.Nphi+g.Nghost)*g.cos_th(g.Nphi+g.Nghost);
                    w_g(i,j)[2] = w_g(i,2*(g.Nphi+g.Nghost)-1-j)[2];
                    w_g(i,j)[3] = w_g(i,2*(g.Nphi+g.Nghost)-1-j)[3] * (-g.cos_th(g.Nphi+g.Nghost)*g.cos_th(g.Nphi+g.Nghost) + g.sin_th(g.Nphi+g.Nghost)*g.sin_th(g.Nphi+g.Nghost))
                                    + 2.*w_g(i,2*(g.Nphi+g.Nghost)-1-j)[1]*g.sin_th(g.Nphi+g.Nghost)*g.cos_th(g.Nphi+g.Nghost);
                }
            }        

            if (i>=g.NR+g.Nghost) {
                if (bound & BoundaryFlags::open_R_outer) {
                    if (w_g(g.NR+g.Nghost-1,j).v_R > 0.) {
                        w_g(i,j) = w_g(g.NR+g.Nghost-1,j);
                    }
                    else {
                        w_g(i,j) = w_g(g.NR+g.Nghost-1,j);
                        w_g(i,j).v_R *= -1 ;
                    }
                }
                else if (bound & BoundaryFlags::set_ext_R_outer) {} //set externally (e.g. inflow)
                else {
                    w_g(i,j) = w_g(g.NR+g.Nghost-1,j);
                    w_g(i,j).v_R *= -1 ;
                }
            }    
                
            if (j < g.Nghost) {
                if (bound & BoundaryFlags::open_Z_inner) {  
                    if (w_g(i,g.Nghost).v_R*g.face_normal_Z(i,g.Nghost).R + 
                        w_g(i,g.Nghost).v_Z*g.face_normal_Z(i,g.Nghost).Z < 0.) {

                        w_g(i,j) = w_g(i,g.Nghost);
                    }
                    else {
                        w_g(i,j)[0] = w_g(i,2*g.Nghost-1-j)[0];
                        w_g(i,j)[1] = w_g(i,2*g.Nghost-1-j)[1] * (g.cos_th(g.Nghost)*g.cos_th(g.Nghost) - g.sin_th(g.Nghost)*g.sin_th(g.Nghost)) 
                                        + 2.*w_g(i,2*g.Nghost-1-j)[3]*g.sin_th(g.Nghost)*g.cos_th(g.Nghost);
                        w_g(i,j)[2] = w_g(i,2*g.Nghost-1-j)[2];
                        w_g(i,j)[3] = w_g(i,2*g.Nghost-1-j)[3] * (-g.cos_th(g.Nghost)*g.cos_th(g.Nghost) + g.sin_th(g.Nghost)*g.sin_th(g.Nghost))
                                    + 2.*w_g(i,2*g.Nghost-1-j)[1]*g.sin_th(g.Nghost)*g.cos_th(g.Nghost);                         
                    }
                    
                }
                else if (bound & BoundaryFlags::set_ext_Z_inner) {} //set externally (e.g. inflow)
                else {  
                    w_g(i,j)[0] = w_g(i,2*g.Nghost-1-j)[0];
                    w_g(i,j)[1] = w_g(i,2*g.Nghost-1-j)[1] * (g.cos_th(g.Nghost)*g.cos_th(g.Nghost) - g.sin_th(g.Nghost)*g.sin_th(g.Nghost)) 
                                    + 2.*w_g(i,2*g.Nghost-1-j)[3]*g.sin_th(g.Nghost)*g.cos_th(g.Nghost);
                    w_g(i,j)[2] = w_g(i,2*g.Nghost-1-j)[2];
                    w_g(i,j)[3] = w_g(i,2*g.Nghost-1-j)[3] * (-g.cos_th(g.Nghost)*g.cos_th(g.Nghost) + g.sin_th(g.Nghost)*g.sin_th(g.Nghost))
                                    + 2.*w_g(i,2*g.Nghost-1-j)[1]*g.sin_th(g.Nghost)*g.cos_th(g.Nghost);
                    // w(i,j,k)[1] = w(i,2*g.Nghost-1-j,k)[1];
                    // w(i,j,k)[2] = w(i,2*g.Nghost-1-j,k)[2];
                    // w(i,j,k)[3] = -w(i,2*g.Nghost-1-j,k)[3];
                }
            }    
        }
    }
}

// compute the conserved quantities (density and momentum) from the primative quantities (density and velocity)
__global__
void _calc_conserved(GridRef g, FieldRef<Quants> q, FieldRef<Prims> w) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {  
            q(i,j).rho = w(i,j).rho;
            q(i,j).mom_R = w(i,j).v_R * w(i,j).rho;
            q(i,j).amom_phi = w(i,j).v_phi * w(i,j).rho * g.Rc(i);
            q(i,j).mom_Z = w(i,j).v_Z * w(i,j).rho;
        }
    }
}

// construct the fluxes based on the velociteis, Roe averaged velocity, and velocities normal to cell faces
__device__ __host__ inline
Quants construct_fluxes(double v_l, double v_r, double v_av, double w_l[4], double w_r[4]) {

    if (v_l <= 0 && v_r >= 0) {
        return {0.,0.,0.,0.};
    }
    else {

        if (v_av > 0.) {   
            double m_l = w_l[0] * v_l ;
            return {m_l, w_l[1] * m_l, w_l[2] * m_l, w_l[3] * m_l} ;
        }

        else if (v_av < 0.) {     
            double m_r = w_r[0] * v_r ;
            return {m_r, w_r[1] * m_r, w_r[2] * m_r, w_r[3] * m_r} ;
        }

        else if (v_av == 0.) {
            double m_l = w_l[0] * v_l ;
            double m_r = w_r[0] * v_r ;
            return {0.5*(       m_l +        m_r), 0.5*(w_l[1]*m_l + w_r[1]*m_r), 
                    0.5*(w_l[2]*m_l + w_r[2]*m_r), 0.5*(w_l[3]*m_l + w_r[3]*m_r)};
        }
    }
}

// compute the radial donor cell gas flux
__device__ __host__ inline
void gas_fluxR(GridRef& g, FieldConstRef<Prims>& w_g, int i, int j, FieldRef<Quants>& fluxR) {

    double normR = g.face_normal_R(i,j).R;
    double normZ = g.face_normal_R(i,j).Z;

    double w_l[4] = {w_g(i-1,j).rho, w_g(i-1,j).v_R, w_g(i-1,j).v_phi, w_g(i-1,j).v_Z};
    double w_r[4] = {w_g(i,j).rho, w_g(i,j).v_R, w_g(i,j).v_phi, w_g(i,j).v_Z};     

    w_l[2] *= g.Re(i) ;
    w_r[2] *= g.Re(i) ;

    double v_l, v_r ;

    v_l = w_l[1] * normR + w_l[3] * normZ;
    v_r = w_r[1] * normR + w_r[3] * normZ; 

    double rhorat = std::sqrt(w_r[0]/w_l[0]);
    double v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    fluxR(i,j) = construct_fluxes(v_l, v_r, v_av, w_l, w_r);
}

// compute the vertical donor cell gas flux
__device__ __host__ inline
void gas_fluxZ(GridRef& g, FieldConstRef<Prims>& w_g, int i, int j, FieldRef<Quants>& fluxZ) {

    double normR = g.face_normal_Z(i,j).R;
    double normZ = g.face_normal_Z(i,j).Z;

    double w_l[4] = {w_g(i,j-1).rho, w_g(i,j-1).v_R, w_g(i,j-1).v_phi, w_g(i,j-1).v_Z};
    double w_r[4] = {w_g(i,j).rho, w_g(i,j).v_R, w_g(i,j).v_phi, w_g(i,j).v_Z};  

    w_l[2] *= g.Rc(i) ;
    w_r[2] *= g.Rc(i) ;

    double v_l, v_r ;
    
    v_l = w_l[1] * normR + w_l[3] * normZ;
    v_r = w_r[1] * normR + w_r[3] * normZ; 

    double rhorat = std::sqrt(w_r[0]/w_l[0]);
    double v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    fluxZ(i,j) = construct_fluxes(v_l, v_r, v_av, w_l, w_r);
}

// compute the donor cell gas flux
__global__ void _calc_donor_flux(GridRef g, FieldConstRef<Prims> w_gas,
                                FieldRef<Quants> fluxR, FieldRef<Quants> fluxZ) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost+1; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost+1; j+=jstride) {
            gas_fluxR(g, w_gas, i, j, fluxR);
            gas_fluxZ(g, w_gas, i, j, fluxZ); 
        }
    }

}

// compute the primative quantities from the conserved quantities
__global__
void _calc_prim(GridRef g, FieldRef<Quants> q, FieldRef<Prims> w) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            w(i,j).rho = q(i,j).rho;
            w(i,j).v_R = q(i,j).mom_R/q(i,j).rho;
            w(i,j).v_phi = q(i,j).amom_phi/(q(i,j).rho * g.Rc(i));
            w(i,j).v_Z = q(i,j).mom_Z/q(i,j).rho;
        }
    }
}

// update conserved quantities from the fluxes
__global__ void _update_quants(GridRef g, FieldRef<Quants> q_mids, FieldRef<Quants> q, double dt,
                                        FieldRef<Quants> fluxR, FieldRef<Quants> fluxZ) {
    
    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) { 
            for (int k=0; k<4; k++) {
                double df = (fluxR(i,j)[k] * g.area_R(i,j) - fluxR(i+1,j)[k] * g.area_R(i+1,j)) 
                        + (fluxZ(i,j)[k] * g.area_Z(i,j) - fluxZ(i,j+1)[k] * g.area_Z(i,j+1));
                q_mids(i,j)[k] = q(i,j)[k] + (dt/g.volume(i,j))*df;
            }
        }
    }
}

// compute the radial van leer gas flux
__device__ __host__ inline
void gas_flux_vlR(GridRef& g, FieldConstRef<Prims>& w_g, int i, int j, FieldRef<Quants>& fluxR) {

    double normR = g.face_normal_R(i,j).R;
    double normZ = g.face_normal_R(i,j).Z;
    double dR_l = g.Re(i)-g.Rc(i-1);
    double dR_r = g.Re(i)-g.Rc(i);

    double w_l[4] = {w_g(i-1,j).rho + vl_R(g,w_g,i-1,j,0)*dR_l, w_g(i-1,j).v_R + vl_R(g,w_g,i-1,j,1)*dR_l, 
                w_g(i-1,j).v_phi + vl_R(g,w_g,i-1,j,2)*dR_l, w_g(i-1,j).v_Z + vl_R(g,w_g,i-1,j,3)*dR_l};

    double w_r[4] = {w_g(i,j).rho + vl_R(g,w_g,i,j,0)*dR_r, w_g(i,j).v_R + vl_R(g,w_g,i,j,1)*dR_r, 
                w_g(i,j).v_phi + vl_R(g,w_g,i,j,2)*dR_r, w_g(i,j).v_Z + vl_R(g,w_g,i,j,3)*dR_r};

    w_l[2] *= g.Re(i) ;
    w_r[2] *= g.Re(i) ;
    
    double v_l, v_r;

    v_l = w_l[1] * normR + w_l[3] * normZ;
    v_r = w_r[1] * normR + w_r[3] * normZ; 

    double rhorat = std::sqrt(w_r[0]/w_l[0]);
    double v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    fluxR(i,j) = construct_fluxes(v_l, v_r, v_av, w_l, w_r);
}

// compute the vertical van leer gas flux
__device__ __host__ inline
void gas_flux_vlZ(GridRef& g, FieldConstRef<Prims>& w_g, int i, int j, FieldRef<Quants>& fluxZ) {

    double normR = g.face_normal_Z(i,j).R;
    double normZ = g.face_normal_Z(i,j).Z;
    double dZ_l = g.Ze(i,j)-g.Zc(i,j-1);
    double dZ_r = g.Ze(i,j)-g.Zc(i,j);

    double w_l[4] = {w_g(i,j-1).rho + vl_Z(g,w_g,i,j-1,0)*dZ_l, w_g(i,j-1).v_R + vl_Z(g,w_g,i,j-1,1)*dZ_l, 
                w_g(i,j-1).v_phi + vl_Z(g,w_g,i,j-1,2)*dZ_l, w_g(i,j-1).v_Z + vl_Z(g,w_g,i,j-1,3)*dZ_l};

    double w_r[4] = {w_g(i,j).rho + vl_Z(g,w_g,i,j,0)*dZ_r, w_g(i,j).v_R + vl_Z(g,w_g,i,j,1)*dZ_r, 
                w_g(i,j).v_phi + vl_Z(g,w_g,i,j,2)*dZ_r, w_g(i,j).v_Z + vl_Z(g,w_g,i,j,3)*dZ_r};


    w_l[2] *= g.Rc(i) ;
    w_r[2] *= g.Rc(i) ;
 
    double v_l, v_r;

    v_l = w_l[1] * normR + w_l[3] * normZ;
    v_r = w_r[1] * normR + w_r[3] * normZ; 

    double rhorat = std::sqrt(w_r[0]/w_l[0]);
    double v_av = (v_l + rhorat * v_r) / (1. + rhorat);

    // Construct fluxes depending on sign of interface velocities

    fluxZ(i,j) = construct_fluxes(v_l, v_r, v_av, w_l, w_r);
}

// compute the van leer gas flux
__global__ void _calc_flux_vl(GridRef g, FieldConstRef<Prims> w_gas, FieldRef<Quants> fluxR, FieldRef<Quants> fluxZ) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost+1; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost+1; j+=jstride) {
                gas_flux_vlR(g, w_gas, i, j, fluxR);
                gas_flux_vlZ(g, w_gas, i, j, fluxZ); 
        }
    }

}

// compute the flux at the boundary cells
__global__ void _set_boundary_flux(GridRef g, int bound, FieldRef<Quants> fluxR, FieldRef<Quants> fluxZ) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx; i<g.NR+2*g.Nghost; i+=istride) {
        for (int j=jidx; j<g.Nphi+2*g.Nghost; j+=jstride) {
            
            if (i <= g.Nghost) {
                if (bound & BoundaryFlags::open_R_inner) {  //outflow
                    if (fluxR(i,j).rho > 0) // prevent inflow
                        fluxR(i,j) = {0.,0.,0.,0.};
                }
                else if (bound & BoundaryFlags::set_ext_R_inner) {} //set externally (e.g. inflow)
                else {  //reflecting
                    fluxR(i,j) = {0.,0.,0.,0.};
                }
            }

            if (j>=g.Nphi+g.Nghost) {
                if (bound & BoundaryFlags::open_Z_outer) {
                    if (fluxZ(i,j).rho < 0) // prevent inflow
                        fluxZ(i,j) = {0.,0.,0.,0.};
                }
                else if (bound & BoundaryFlags::set_ext_Z_outer) {} //set externally (e.g. inflow)
                else {
                    fluxZ(i,j) = {0.,0.,0.,0.};
                }
            }        

            if (i>=g.NR+g.Nghost) {
                if (bound & BoundaryFlags::open_R_outer) {
                    if (fluxR(i,j).rho < 0) // prevent inflow
                        fluxR(i,j) = {0.,0.,0.,0.};
                }
                else if (bound & BoundaryFlags::set_ext_R_outer) {} //set externally (e.g. inflow)
                else {
                    fluxR(i,j) = {0.,0.,0.,0.};
                }
            }    
            
            if (j <= g.Nghost) {
                if (bound & BoundaryFlags::open_Z_inner) {  
                    if (fluxZ(i,j).rho > 0) // prevent inflow
                        fluxZ(i,j) = {0.,0.,0.,0.};
                }
                else if (bound & BoundaryFlags::set_ext_Z_inner) {} //set externally (e.g. inflow)
                else {  
                    fluxZ(i,j) = {0.,0.,0.,0.};
                }
            }
            
        }
    }

}

// gas evolution operator
void GasDynamics::operator() (Grid& g, Field<Prims>& w_gas, const double* nu, double dt) {
    if (g.Nghost < 2)
        throw std::invalid_argument("Gas dynamics requires at least 2 ghost cells") ;

    Field<Quants> q_mids = Field<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost);
    Field<Quants> q = Field<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost);

    Field<Quants> fluxR = Field<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost);
    Field<Quants> fluxZ = Field<Quants>(g.NR+2*g.Nghost, g.Nphi+2*g.Nghost);

    dim3 threads(16,8,4) ;
    dim3 blocks((g.NR + 2*g.Nghost+15)/16,(g.Nphi + 2*g.Nghost+7)/8, 1) ;
    //dim3 blocks(4,4,4) ;

    _set_boundaries<<<blocks,threads>>>(g, w_gas, _boundary, _floor);
    check_CUDA_errors("_set_boundaries") ;
    _calc_conserved<<<blocks,threads>>>(g, q, w_gas);
    check_CUDA_errors("_calc_conserved") ;

    // Calc donor cell flux
    _calc_donor_flux<<<blocks,threads>>>(g, w_gas, fluxR, fluxZ);
    check_CUDA_errors("_calc_donor_flux") ;
    
    // Update quantities a half time step and and source terms.
    _set_boundary_flux<<<blocks,threads>>>(g, _boundary, fluxR, fluxZ);
    check_CUDA_errors("_set_boundary_flux") ;
    _update_quants<<<blocks,threads>>>(g, q_mids, q, dt/2., fluxR, fluxZ);
    check_CUDA_errors("_update_quants") ;
    sources_gas(g, w_gas, q_mids, nu, _cs, _Mstar, _floor, dt*0.5);
    _calc_prim<<<blocks,threads>>>(g, q_mids, w_gas);
    check_CUDA_errors("_calc_prim") ; 
    
    _set_boundaries<<<blocks,threads>>>(g, w_gas, _boundary, _floor);
    check_CUDA_errors("_set_boundaries") ;

    // Compute fluxes with Van Leer
    _calc_flux_vl<<<blocks,threads>>>(g, w_gas, fluxR, fluxZ);
    check_CUDA_errors("_calc_diff_flux_vl") ;

    // Update quantities a full time step and and source terms.

    _set_boundary_flux<<<blocks,threads>>>(g, _boundary, fluxR, fluxZ);
    check_CUDA_errors("_set_boundary_flux") ;
    // set_flux_to_zero<<<blocks,threads>>>(g, fluxR);
    _update_quants<<<blocks,threads>>>(g, q_mids, q, dt, fluxR, fluxZ);
    check_CUDA_errors("_update_quants") ;
    //sources_gas(g, w_gas, q_mids, nu, _cs, _Mstar, _floor, dt);
    _calc_prim<<<blocks, threads>>>(g, q_mids, w_gas);
    check_CUDA_errors("_calc_prim") ; 
}

// compute the CFL timestep for each grid point
__global__
void _compute_CFL_diff(GridRef g, FieldConstRef<Prims> w_gas, FieldRef<double> CFL_grid,
                        double CFL_adv) {

    int iidx = threadIdx.x + blockIdx.x*blockDim.x ;
    int jidx = threadIdx.y + blockIdx.y*blockDim.y ;
    int istride = gridDim.x * blockDim.x ;
    int jstride = gridDim.y * blockDim.y ;

    for (int i=iidx+g.Nghost; i<g.NR+g.Nghost; i+=istride) {
        for (int j=jidx+g.Nghost; j<g.Nphi+g.Nghost; j+=jstride) {
            double CFL_k = 1e308;

            double dtR = abs(g.dRe(i)/w_gas(i,j).v_R);
            double dtZ = abs(g.dZe(i,j)/w_gas(i,j).v_Z);

            double CFL_RZmin = min(dtR, dtZ);
            CFL_k = min(CFL_k, CFL_adv*CFL_RZmin);

            CFL_grid(i,j) = CFL_k;
        }
    } 
}

// wrapper function to compute the CFL timestep
double GasDynamics::get_CFL_limit(const Grid& g, const Field<Prims>& w_gas) {

    dim3 threads(32,32) ;
    dim3 blocks((g.NR + 2*g.Nghost+31)/32,(g.Nphi + 2*g.Nghost+31)/32) ;

    Field<double> CFL_grid = create_field<double>(g);
    set_all(g, CFL_grid, std::numeric_limits<double>::max());

    _compute_CFL_diff<<<blocks,threads>>>(g, w_gas, CFL_grid, _CFL_adv);
    check_CUDA_errors("_compute_CFL_diff") ;
    Reduction::scan_R_min(g, CFL_grid);

    double dt = CFL_grid(g.NR+g.Nghost-1,g.Nghost) ;
    for (int j=g.Nghost; j < g.Nphi+g.Nghost; j++) {
        dt = std::min(dt, CFL_grid(g.NR+g.Nghost-1, j)) ;
    }

    return dt;
}
