
#ifndef _CUDISC_GASDYNAMICS_H_
#define _CUDISC_GASDYNAMICS_H_

#include "cuda_array.h"
#include "field.h"
#include "flags.h"
#include "grid.h"
#include "utils.h"
#include "dustdynamics.h"
#include "sources.h"

__global__
void _set_boundaries(GridRef g, Field3DRef<Prims> w, int bound, double floor) ;

class GasDynamics {

    public:

        GasDynamics(FieldConstRef<double> cs, double CFL_adv=0.4, double CFL_diff=0.1, double Mstar = 1.0, double floor=1.e-30) : 
                _cs(cs), _CFL_adv(CFL_adv), _CFL_diff(CFL_diff), _floor(floor) {};

        void set_CFL_adv(double cfl) {
            _CFL_adv = cfl;
        }

        void set_CFL_diff(double cfl) {
            _CFL_diff = cfl;
        }

        void set_boundaries(int flag) {
            _boundary = flag ;
        }
        int get_boundaries() const {
        return _boundary ;
        }

        void floor_above(Grid&g, Field<Prims>& w_dust, Field<Prims>& w_gas, CudaArray<double>& h);

        void operator() (Grid& g, Field<Prims>& w_gas, const double* nu, double dt) ;

        double get_CFL_limit(const Grid& g, const Field<Prims>& w_gas) ;
        double get_CFL_limit_debug(const Grid& g, const Field<Prims>& w_gas);
        // double get_CFL_limit_debug(const Grid& g, const Field3D<Quants>& q, const Field3D<double>& D) ;

    private:

        FieldConstRef<double> _cs;
        double _CFL_adv;
        double _CFL_diff;
        double _floor;
        double _Mstar;

        int _boundary = BoundaryFlags::open_R_inner | BoundaryFlags::open_R_outer;

} ;


#endif
