// FVM_Burgers_1D.cpp :
//
#include <iostream>
#include <fstream>
#define _USE_MATH_DEFINES
#include <math.h>

using namespace std;

/**
*   -- Viscous Burgers equation --

    ∂u/∂t + 0.5*∂u^2/∂x = ν * ∂^2(u)/∂x^2,  u = u(x,t), -L ≤ x ≤ L, 0≤ t ≤T,
    u(x, t=0) = u_0(x)   - initial condition
	u(L, t) = u(-L,t) – periodic bc

    - Formulas for different numerical schemes

    * 1-st order explicit Euler scheme:

    u(i,n+1) = u(i,n) - dt/dx_i * (F(i+1/2,n,*) - F(i-1/2,n,*) - ν(Q(i+1/2, n) - Q(i-1/2,n) ))

    * 2-nd order explicit Runge-Kutta scheme:

    u(i,n+1/2) = u(i,n) - 0.5 dt/dx_i (F(i+1/2,n,*) - F(i-1/2, n,*) - ν (Q(i+1/2,n) - Q(i-1/2, n ))
    u(i,n+1) = u(i,n) - dt/dx_i (F(i+1/2,n+1/2,*) - F(i-1/2,n+1/2,*) - ν (Q(i+1/2,n+1/2) - Q(i-1/2,n+1/2)))

    * Implicit Euler scheme:

    u(i,n+1) = u(i,n) - ∆t/dx_i (F(i+1/2,n+1,*) - F(i-1/2,n+1,*) - ν (Q(i+1/2, n+1)-Q(i-1/2, n+1) ))

	* Semi-implicit scheme after using trapezoidal rule (Crank–Nicolson method):

    u_i^(n+1)=u_i^n-0.5 ∆t/(Δx_i ) (F_(i+1/2)^(n,*)-F_(i-1/2)^(n,*) - ν(Q_(i+1/2)^n-Q_(i-1/2)^n )+ F_(i+1/2)^(n+1,*)-F_(i-1/2)^(n+1,*) - ν(Q_(i+1/2)^(n+1)-Q_(i-1/2)^(n+1) ))

    *
    *                   Helpful formulas with slope limiters case
    *
    F(i±1/2, *) = 0.5([F(u(i±1/2, L) + F(u(i±1/2, R)] - a(i±1/2)([u(i±1/2,R) - u(i±1/2,L)])
    a(i±1/2) = max[ρ(∂F(u(i±1/2,L))/∂u),ρ(∂F(u(i±1/2,R))/∂u)]

    -  piecewise linear representation
    u_(i+1/2)^L =  u_i+0.5φ(r_i )(u_(i+1)-u_i ),           u_(i+1/2)^R = u_(i+1)-0.5φ(r_(i+1))(u_(i+2)-u_(i+1) )
    u_(i-1/2)^L =  u_(i-1)+0.5φ(r_(i-1) )(u_i-u_(i-1) ),   u_(i-1/2)^R = u_i-0.5φ(r_i)(u_(i+1)-u_i ),

    where r_i=((u_i-u_(i-1) ))/((u_(i+1)-u_i ) )

    -  minmod   limiter  : φ(r) = max(0, min(1,r))
    -  superbee limiter  : φ(r) = max(0, min(1,2r), min(r,2))
    -  van Albada limiter: φ(r) = (r^2 + r)/(r^2 + 1)
    -  van Leer limiter  : φ(r) = (r + |r|)/(|r| + 1)

    -----------------------------------------------------------------------------------------------------------------------------------
    * Implicit Euler schemes :
    - 1) variant of linearization -> finite difference time derivative for term {du/dt ~ (u(i,n+1) - u(i,n))/dt}
        u(i,(n+1)) = u(i,n) - 2∆t/(Δx_i + Δx_(i-1)) (
            [dF/du](i+1/2, n*) (u(i+1,n+1) + u(i,n+1)) - [dF/du](i-1/2, n*) (u(i,n+1) + u(i-1,n+1)) +
            (F(i+1/2, n*) - [dF/du](i+1/2, n*) u(i+1/2, n*)) - (F(i-1/2, n*) - [dF/du](i-1/2, n*) u(i-1/2, n*) )
            - ν((u(i+1, n+1) - u(i,n+1))/Δx_i - (u(i,n+1)-u(i-1,n+1))/Δx(i-1)));  (iv)

    - 2) variant of linearization -> {du/dt = L(u, u_x) = -dF/dx + dQ/dx -> Finite differemce approx of L in points x(i±1/2, n)}
        u(i,n+1) = u(i,n) - 2∆t/(Δx(i)+Δx(i-1)) {
            F(i+1/2,n*) + ∆t[dF/du](i+1/2, n*) [(Q(i+1,n) - Q(i,n))/Δx(i) -(F(i+1,n)-F(i,n))/Δx(i)]
            - (F(i-1/2, n*) + ∆t[dF/du](i-1/2, n*) [(Q(i,n) - Q(i-1,n))/Δx(i-1) -(F(i,n) - F(i-1,n))/Δx(i-1)])
            - ν((u(i+1, n+1) - u(i,n+1))/Δx(i) - (u(i,n+1) - u(i-1,n+1))/Δx(i-1) ) }


    * Semi-implicit scheme after using trapezoidal rule (Crank–Nicolson method):
        u(i,n+1) = u(i,n) - 0.5 ∆t/Δx(i){ F(i+1/2, n*) - F(i-1/2, n*) - (Q(i+1/2, n) - Q(i-1/2,n)) + F(i+1/2, n+1,*) - F(i-1/2,n+1,*) - (Q(i+1/2, n+1) - Q(i-1/2, n+1)) }

    Further solver system of linear equations (with 5-diagonal matrix) regarding u(i, n+1)  for i = {0,1..,Nx}
    Need to solve : A * u(n+1) = b(n), where A - matrix {(Nx+1) x (Nx + 1)}
        |d3 d4 0 .. 0 d5|
        |d2 d3 d4 0....0|
    A = |0 d2 d3 d4 0..0|  due periodic bc 3-diagonal system (in case with Dirichler/Neumann bc) transform to 5 diagonal
        |...............|
        |0....0 d2 d3 d4|
        |d1 0....0 d2 d3|

*/

// types of numerical scheme
enum T_SCHEME
{
    // Euler forward/backward
    F_EULER = 0,
    B_EULER = 1,
    // Runge-Kutta schemes
    RK2 = 2,
    RK3 = 3,
    RK4 = 4,
    CN = 5, // Crank–Nicolson
    CN_NR = 6// Crank–Nicolson with Newton-Raphson solver
};

// type of linearization formula
// for implicit schemes
enum T_LINEAR
{
    NO = 0,
    FINITE_DIFF_DU_DT = 1,
    DU_DT_EQUAL_RHS = 2
};

/**
*               /               CELL-i              \
*               |                                   |
*  in code      |  uL[i]          u[i]     uR[i]    |
*               |  fL[i]           ^       /fR[i]/  |
*               |  q[i]            |                |
*  in formulas  | uR(i-1/2)        |       uL(i+1/2)|
*               | F(i-1/2,*)       |      F(i+1/2,*)|
*               | Q(i-1/2)*dx      |                |
*               |     ^            |          ^     |
*               |     |            |          |     |
*               |     |            |          |     |
*               | x[i-1/2]        x[i]     x[i+1/2] |
*               |                                   |
*               \               CELL-i              /
**/

/** ----------- Slope limiter functions
    r = q[i] / q[i + 1], q[i] = u[i]-u[i-1]
    - minmod   limiter : φ(r) = max(0, min(1, r))
    - superbee limiter : φ(r) = max(0, min(1, 2r), min(r, 2))
    - van Albada limiter : φ(r) = (r ^ 2 + r) / (r ^ 2 + 1)
    - van Leer limiter : φ(r) = (r + | r | ) / (| r | + 1)

 */

double no_limiters(double q1, double q2)
{
    return 0.0;
}

double minmod(double q1, double q2)
{
    // r = q[i] / q[i + 1], q[i] = u[i]-u[i-1]
    if (q2 != 0.0)return max(0.0, min(1.0, q1 / q2));
    if (q1 != 0.0)return 1.0;
    return 0.0;
}

double superbee(double q1, double q2)
{
    if (q2 != 0.0) {
        double r = q1 / q2;
        return max(0.0, max(min(1.0, 2.0*r), min(r, 2.0)));
    }
    if (q1 != 0.0)return 2.0;
    return 0.0;
}

double vanAlbada(double q1, double q2)
{
    if (q2 != 0.0) {
        double r = q1 / q2;
        return (r*r + r) / (r*r + 1.0);
    }
    if (q1 != 0.0)return 1.0;
    return 0.0;
}

double vanLeer(double q1, double q2)
{
    if (q2 != 0.0) {
        double r = q1 / q2;
        return (r + abs(r)) / (abs(r) + 1.0);
    }
    if (q1 != 0.0)return 2.0;
    return 0.0;
}

// initial condition functions
double tophat(double x, double min_x, double max_x)
{
    double center = 0.5 * (min_x + max_x);
    double _dx = 0.5 * (max_x - min_x) / 4.0;
    if (x < center - _dx || x > center + _dx)return 0.0;
    return 1.0;
}

double sine(double x, double min_x, double max_x)
{
    // number of waves
    int nw = 2;
    return 0.5 + 0.5 * sin(nw*(2.*(x - min_x)/(max_x - min_x)-1.0)*M_PI);
}

// mesh distribution
double uniform(double x)
{
    return x;
}

// x[0,1] -> f(x)[0,1]
double rare_in_center(double x)
{
    if (x < 0.5) return x*sqrt(2*x);
    return 1.0 - abs(x - 1.0) * sqrt(2 * abs(x - 1.0));
}

// function of nonlinear convective term
double Fu(double u)
{
    return 0.5 * u * u;
    //return u;
}

// derivative of nonlinear term
double dFu(double u)
{
    return u;
    //return 1;
}

// second order derivative of nonlinear term for implicit numerical schemes
double d2Fu(double u)
{
    return 1;
    //return 0;
}

class FV_Burgers1D
{
public:
    int max_steps, nx, ng;
    double min_x, max_x;// dx;
    double nu, dt, stop_time;
    double (*phi)(double, double); // pointer to slope limiter function
    double (*ic_fun)(double, double, double); // pointer to initial condition function
    double (*mapping)(double);// pointer to mesh function

    //Cell data
    double *x, *u, *u_new, *uL, *uR;
    double *fL, *q;
    // help vars for RK4
    double* k_tmp, * k_tmp2;

    ofstream output;

    // data for implicit schemes which use LU-decomposition
    double *L, *D, *U;// 3 - diagonal and rhs of SLAE data; *b -> u_new
    double* p, * r, *dfL;// help variables for

    // For Newton-Raphson
    double* fL_new, * q_new , *du;

    FV_Burgers1D(int n=100,  double xL=-1, double xR= 1, double vis=0.001, double Dt=0.01, double time=2.0,
        double (*limiter)(double, double) = minmod, double (*ic_f)(double, double, double) = tophat, double (*map_mesh)(double) = uniform)
    {
        nx = n; ng = 2;
        stop_time = time;
        dt = Dt;
        max_steps = int(stop_time / dt + 0.5);
        min_x = xL; max_x = xR;
        nu = vis;
        dt = stop_time / max_steps;
        phi = limiter;
        ic_fun = ic_f;
        mapping = map_mesh;

        data_alloc();
        init_mesh();
        initial_conditions();
    }

    void data_alloc()
    {
        //cell = new Cell[nx + 2 * ng];
        x = new double[nx + 2 * ng];
        u = new double[nx + 2 * ng];
        u_new = new double[nx + 2 * ng];
        uL = new double[nx + 2 * ng];
        uR = new double[nx + 2 * ng];
        fL = new double[nx + 2 * ng];
        q = new double[nx + 2 * ng];
    }

    void init_mesh()
    {
        double L = max_x - min_x;
        for (size_t i = 0; i <= nx; i++)
        {
            x[i + ng] = min_x + L * mapping(double(i) / nx);
            //printf("%f\n", x[i]);
        }

        // extrapolate to ghost nodes
        for (size_t i = 0; i < ng; i++) {
            x[i] = x[i + nx] - L;
            x[i + ng + nx] = x[i + ng] + L;
        }
    }
    void initial_conditions()
    {
        for (int i = 0; i < nx + 2*ng; i++) {
            u[i] = u_new[i] = ic_fun(x[i], min_x, max_x);
        }
    }

    void update_ghost(double *_u)
    {
        for (size_t i = 0; i < ng; i++) {
            _u[i] = _u[i + nx];
            _u[i + ng + nx] = _u[i + ng];
        }
    }
    void reconstruct(double *_u, bool only_uLR = false, bool is_implicit = false)
    {
        for (size_t i = ng-1; i <= nx + ng + 1; i++)
            q[i] = _u[i] - _u[i - 1];
        for (size_t i = ng-1; i <= nx + ng; i++) {
            // r_i = q[i] / q[i+1]
            // du = 0.5φ(r_i)(u_(i+1)-u_i)
            // uL[i] -> u(i-1/2, R) = u_i - du ; uR[i] -> u(i + 1/2, L) = u_i + du
            double du = 0.5 * phi(q[i], q[i+1]) * q[i+1];
            uL[i] = _u[i] - du;
            uR[i] = _u[i] + du;
        }
        /*

        F(i ± 1/2, *) = 0.5([F(u(i ± 1/2, L) + F(u(i ± 1/2, R)] - a(i ± 1/2)([u(i ± 1/2, R) - u(i ± 1/2, L)])
        a(i ± 1/2) = max[ρ(∂F(u(i ± 1/2, L)) / ∂u), ρ(∂F(u(i ± 1/2, R)) / ∂u)]

        */
        if (!only_uLR) {
            for (size_t i = ng; i <= nx + ng; i++) {
                fL[i] = 0.5 * (Fu(uR[i - 1]) + Fu(uL[i]) - max(abs(dFu(uR[i - 1])), abs(dFu(uL[i]))) * (uL[i] - uR[i - 1]));
            }

            /*  for implicit schemes we need in
            *    [dF/du](i+1/2, n,*) = 0.5(dF/du(u(i±1/2, L) + dF/du (u(i±1/2, R) - a(i±1/2)' [u(i±1/2,R) - u(i±1/2,L)]) , (dfa)
            *    a(i±1/2)' = max[ρ(∂^2 F(u(i±1/2, L))/∂u^2), ρ(∂^2 F(u(i±1/2,R))/∂u^2)]
            *    u(i+1/2, n*) = u(i+1/2,L)   (same idea as for F_(i+1/2)^(n,*) )   (uL)
            */
            if (is_implicit) {
                for (size_t i = ng; i <= nx + ng; i++) {
                    dfL[i] = 0.5 * (dFu(uR[i - 1]) + dFu(uL[i]) - max(abs(d2Fu(uR[i - 1])), abs(d2Fu(uL[i]))) * (uL[i] - uR[i - 1]));
                }
            }
        }
    }

    // new reconstruct for NR implicit
    void reconstruct(double* _u, double *_q, double *_fL, double *_dfL, bool only_uLR = false, bool is_implicit = false)
    {
        for (size_t i = ng - 1; i <= nx + ng + 1; i++)
            _q[i] = _u[i] - _u[i - 1];
        for (size_t i = ng - 1; i <= nx + ng; i++) {
            // r_i = q[i] / q[i+1]
            // du = 0.5φ(r_i)(u_(i+1)-u_i)
            // uL[i] -> u(i-1/2, R) = u_i - du ; uR[i] -> u(i + 1/2, L) = u_i + du
            double _du = 0.5 * phi(_q[i], _q[i + 1]) * _q[i + 1];
            uL[i] = _u[i] - _du;
            uR[i] = _u[i] + _du;
        }
        /*

        F(i ± 1/2, *) = 0.5([F(u(i ± 1/2, L) + F(u(i ± 1/2, R)] - a(i ± 1/2)([u(i ± 1/2, R) - u(i ± 1/2, L)])
        a(i ± 1/2) = max[ρ(∂F(u(i ± 1/2, L)) / ∂u), ρ(∂F(u(i ± 1/2, R)) / ∂u)]

        */
        if (!only_uLR) {
            for (size_t i = ng; i <= nx + ng; i++) {
                _fL[i] = 0.5 * (Fu(uR[i - 1]) + Fu(uL[i]) - max(abs(dFu(uR[i - 1])), abs(dFu(uL[i]))) * (uL[i] - uR[i - 1]));
            }

            /*  for implicit schemes we need in
            *    [dF/du](i+1/2, n,*) = 0.5(dF/du(u(i±1/2, L) + dF/du (u(i±1/2, R) - a(i±1/2)' [u(i±1/2,R) - u(i±1/2,L)]) , (dfa)
            *    a(i±1/2)' = max[ρ(∂^2 F(u(i±1/2, L))/∂u^2), ρ(∂^2 F(u(i±1/2,R))/∂u^2)]
            *    u(i+1/2, n*) = u(i+1/2,L)   (same idea as for F_(i+1/2)^(n,*) )   (uL)
            */
            if (is_implicit) {
                for (size_t i = ng; i <= nx + ng; i++) {
                    _dfL[i] = 0.5 * (dFu(uR[i - 1]) + dFu(uL[i]) - max(abs(d2Fu(uR[i - 1])), abs(d2Fu(uL[i]))) * (uL[i] - uR[i - 1]));
                }
            }
        }
    }

    //swap u and u_new
    void swap()
    {
        double *tmp_ptr = u;
        u = u_new, u_new = tmp_ptr;
    }

    // right hand side of numerical scheme
    void RHS(double *_u, double *_F)
    {
        update_ghost(_u);
        reconstruct(_u);
        for (size_t i = ng; i < nx + ng; i++) {
            double dx0 = x[i] - x[i - 1];
            double dx1 = x[i + 1] - x[i];
            double dxi = (dx0 + dx1) / 2.0;
            _F[i] = - (fL[i + 1] - fL[i] - nu * (q[i + 1] / dx1 - q[i] / dx0)) / dxi;
        }
    }

    void explicit_step(double _dt, double *_u_old, double *_u, double *_u_new)
    {
        double u_sum = 0;
        RHS(_u, _u_new);
        for (size_t i = ng; i < nx + ng; i++) {
            _u_new[i] = _u_old[i] + _dt * _u_new[i];
            u_sum += _u_new[i];
        }
        printf("%f\n", u_sum);
    }


    /* Implicit step_solver
    */
    // Newton-Raphson for Crank-Nicolson scheme


    void implicit_step(int t, int type_linearization = T_LINEAR::DU_DT_EQUAL_RHS, bool is_CN = false)
    {
        update_ghost(u);
        reconstruct(u, false, true);

        /************************************************
        *   Formulas from (iv) numerical scheme in doc file
        *   L[i] =-2∆t/(Δx(i) + Δx(i-1)) ([dF/du](i-1/2, n*) + ν/Δx(i-1))
        *   D[i] = 1 + 2∆t/(Δx(i) + Δx(i-1)) {[dF/du](i + 1/2, n*) - [dF/du](i - 1/2, n*) + ν/Δx(i) + ν/Δx(i-1)}
        *   U[i] = 2∆t/(Δx(i) + Δx(i-1) ) ([dF/du](i+1/2, n,*) - ν/Δx(i))
        *   b[i] = u(i,n) - 2∆t/(Δx(i) + Δx(i-1)) { F(i + 1/2, n*) - [dF/du](i + 1/2, n*) u(i + 1/2, n*) - (F(i - 1/2, n*) - [dF/du](i - 1/2, n*) u(i - 1/2, n*)) }

        Where
        *    [dF/du](i+1/2, n,*) = 0.5(dF/du(u(i±1/2, L) + dF/du (u(i±1/2, R) - a(i±1/2)' [u(i±1/2,R) - u(i±1/2,L)]) , (dfa)
        *    a(i±1/2)' = max[ρ(∂^2 F(u(i±1/2, L))/∂u^2), ρ(∂^2 F(u(i±1/2,R))/∂u^2)]
        *    u(i+1/2, n*) = u(i+1/2,L)   (same idea as for F_(i+1/2)^(n,*) )   (uL)
        *
        * ---------------------------------------------------------------------------------------------------------------------
        * u(i+1/2, n*) = 0.5/(Δx(i) + Δx(i-1)) ((u(i+1/2, R) + u(i+1))Δx(i) + (u(i+1/2, L) + u(i))Δx(i-1))   - integrated by 2 slopes trapezes, (uI)

        F(i+1/2, n*) = F { 0.5/(Δx(i) + Δx(i-1)) ((u(i+1/2, R) + u(i+1))Δx(i) + (u(i+1/2, L) + u(i))Δx(i-1)) }, (Fi)
        [dF/du](i+1/2, n*) = dF/du { 0.5/(Δx(i) + Δx(i-1)) ((u(i+1/2, R) + u(i+1))Δx(i) + (u(i+1/2, L) + u(i))Δx(i-1)) },  (dfi).

        * *****************************************************************************************
        *   Formulas from (v) numerical scheme in doc file
        *   L[i] = -2 ν dt/(dx(i-1)(dx(i) + dx(i-1)))
        *   D[i] = 1 + 2ν dt/(dx(i) dx(i-1) )
        *   U[i] = -2 ν dt/(dx(i)(dx(i) + dx(i-1)))
        *   b[i] = u[i] - 2dt/(dxi + dx(i-1)) {
        *                        F(i+1/2, n*) + dt [dF/du](i+1/2, n*) [(Q(i+1,n) - Q(i,n))/dx_i - (F(i+1,n) - F(i,n))/dx_i ]
        *                       -(F(i-1/2, n*) + dt [dF/du](i-1/2, n*) [(Q(i,n) - Q(i-1, n))/dx(i-1) -(F(i,n) - F(i-1, n))/dx(i-1)] ) }
        *   Q(i+1) = 0.5(Q(i+3/2) + Q(i+1/2)), Q(i) = 0.5(Q(i+1/2) + Q(i-1/2))
        *   Q(i+1) - Q(i) = 0.5(Q_(i+3/2)^n-Q_(i-1/2)^n ) = 0.5 * (q[i+1] - q[i-1])
        *   Q(i) - Q(i-1) = 0.5(Q_(i+1/2)^n-Q_(i-3/2)^n ) = 0.5 * (q[i] - q[i-2])
        *
        */
        switch (type_linearization){

            case T_LINEAR::FINITE_DIFF_DU_DT:
                for (size_t i = ng; i < ng + nx; i++) {
                    double dx0 = x[i] - x[i - 1];
                    double dx1 = x[i + 1] - x[i];
                    double dxi = (dx0 + dx1) / 2.0;
                    double dtx = dt / dxi, nu0 = nu / dx0, nu1 = nu / dx1;
                    if (is_CN)dtx *= 0.5;
                    // [dF / du](i - 1 / 2, n*) -> dfL[i], F(i - 1 / 2, n*) -> fL[i], u(i - 1 / 2, n*) -> uL[i]
                    L[i - ng] = -dtx * (dfL[i] + nu0);
                    D[i - ng] = 1.0 + dtx* (dfL[i + 1] - dfL[i] + nu0 + nu1);
                    U[i - ng] = dtx * (dfL[i + 1] - nu1);
                    // used u_new for vector rhs=b
                    u_new[i - ng] = u[i] - dtx * (fL[i + 1] - dfL[i + 1] * uL[i + 1] - (fL[i] - dfL[i] * uL[i]));
                    // additional term for Crank-Nicolson scheme
                    if (is_CN)u_new[i - ng] -= dtx * (fL[i + 1] - fL[i] - nu * (q[i + 1] / dx1 - q[i] / dx0));
                }

                LU_sparse(L, D, U, L, D, U, p, r, nx);
                break;

            case T_LINEAR::DU_DT_EQUAL_RHS:
                for (size_t i = ng; i < ng + nx; i++) {
                    double dx0 = x[i] - x[i - 1];
                    double dx1 = x[i + 1] - x[i];
                    double dxi = (dx0 + dx1) / 2.0;
                    double dtx = dt / dxi, nu0 = nu / dx0, nu1 = nu / dx1;
                    if (is_CN)dtx *= 0.5;
                    // [dF / du](i - 1 / 2, n*) -> dfL[i], F(i - 1 / 2, n*) -> fL[i], u(i - 1 / 2, n*) -> uL[i]
                    // matrix of SLAE dont change along time
                    if (t == 0){
                        L[i - ng] = -dtx * nu0;
                        D[i - ng] = 1.0 + dtx * (nu0 + nu1);
                        U[i - ng] = -dtx * nu1;
                    }
                    // used u_new for vector rhs=b
                    u_new[i - ng] = u[i] - dtx * (fL[i + 1] + dt * dfL[i + 1] * (0.5 * (q[i + 1] - q[i - 1]) - (Fu(u[i + 1]) - Fu(u[i]))) / dx1
                        - (fL[i] + dt * dfL[i] * (0.5 * (q[i] - q[i - 2]) - (Fu(u[i]) - Fu(u[i - 1]))) / dx0));
                    // additional term for Crank-Nicolson scheme
                    if (is_CN)u_new[i - ng] -= dtx * (fL[i + 1] - fL[i] - nu * (q[i + 1] / dx1 - q[i] / dx0));
                }
                // only initially do LU-decomposition
                if (t == 0)LU_sparse(L, D, U, L, D, U, p, r, nx);
                break;
        }

        LU_backward_steps(u_new, L, D, U, p, r, nx);
    }

    void LU_backward_steps(double* b, double* l, double* d, double* _u, double* p, double* r, double *y,  size_t N)
    {
        /*
        *  y[0] = b[0]
        *  y[i] = b[i] - l[i - 1] y[i - 1], i = [1, .., N - 2]
        *  y[N - 1] = b[N - 1] - ∑{k = 0,1,.., N - 3}r[k] * y[k] - l[N - 2]* y[N - 2]
        */

        // use u_new -> y
        y[0] = b[0];
        y[N - 1] = b[N - 1] - r[0] * y[0];
        for (size_t i = 1; i <= N - 2; i++) {
            y[i] = b[i] - l[i - 1] * y[i - 1];
            if (i != N - 2)y[N - 1] -= r[i] * y[i];
            else y[N - 1] -= l[N - 2] * y[N - 2];
        }

        /*
        x[N-1] = y[N-1] / d[N-1]
        x[N-2] = (y[N-2] - x[N-1] *_u[N-2])/d[N-2]

        x[i] = (y[i] - x[i+1] * _u[i] - x[N-1] * p[i]) / d[i], i=[N-3,.,1,0]

        */
        // x[i] ->  y[i]
        y[N - 1] = y[N - 1] / d[N - 1];
        y[N - 2] = (y[N - 2] - y[N - 1] * _u[N - 2]) / d[N - 2];
        for (int i = N - 3; i >= 0; i--) {
            y[i] = (y[i] - y[i + 1] * _u[i] - y[N - 1] * p[i]) / d[i];
        }
    }

    void LU_backward_steps(double* b, double* l, double* d, double* _u, double* p, double* r, size_t N)
    {
        /*
        *  y[0] = b[0]
        *  y[i] = b[i] - l[i - 1] y[i - 1], i = [1, .., N - 2]
        *  y[N - 1] = b[N - 1] - ∑{k = 0,1,.., N - 3}r[k] * y[k] - l[N - 2]* y[N - 2]
        */

        double u_sum = 0;
        // use u_new -> y
        u_new[0] = b[0];
        u_new[N - 1] = b[N - 1] - r[0] * u_new[0];
        for (size_t i = 1; i <= N-2; i++){
            u_new[i] = b[i] - l[i - 1] * u_new[i - 1];
            if (i != N - 2)u_new[N - 1] -= r[i] * u_new[i];
            else u_new[N - 1] -= l[N - 2] * u_new[N - 2];
        }

        /*
        x[N-1] = y[N-1] / d[N-1]
        x[N-2] = (y[N-2] - x[N-1] *_u[N-2])/d[N-2]

        x[i] = (y[i] - x[i+1] * _u[i] - x[N-1] * p[i]) / d[i], i=[N-3,.,1,0]

        */
        // u[ng + i] -> x[i], u_new[i] -> y[i]
        u[ng + N - 1] = u_new[N - 1] / d[N - 1];
        u[ng + N - 2] = (u_new[N - 2] - u[ng + N - 1] * _u[N - 2]) / d[N - 2];
        for (int i = N-3; i >= 0; i--){
            u[ng + i] = (u_new[i] - u[ng + i + 1] * _u[i] - u[ng + N - 1] * p[i]) / d[i];
            u_sum += u[i + ng];
        }
        printf("%f\n", u_sum);
    }
    // band matrix factorization
    void LU_sparse(double *L, double *D, double *U, double *l, double *d,  double *u_, double *p, double *r, size_t N)
    {
        /*
        d[0]=D[0], u[0]=U[0],p[0]=L[0],
        l[0]=L[1]/d[0], r[0]=U[N-1]/d[0]
        for  i=[1,N-2]
            d[i]=D[i] - u[i-1]l[i-1]
            if i < N-2: u[i] = U[i], p[i] =- l[i-1]*p[i-1]
            else u[i] = U[i]-p[i-1]*l[i-1]
            if i < N-2: l[i]=L[i+1]/d[i], r[i]=-u[i-1]*r[i-1]/d[i]
            else l[i]=(L[i+1]-u[i-1]*r[i-1])/d[i]
        end
        d[N-1]=D[N-1] – sum(r[i]*p[i])-u[N-2]*l[N-2]

        */
        d[0] = D[0]; u_[0] = U[0]; p[0] = L[0];
        l[0] = L[1] / d[0]; r[0] = U[N - 1] / d[0];
        d[N - 1] = D[N - 1] - r[0] * p[0];
        for (size_t i = 1; i <= N-2; i++){
            d[i] = D[i] - u_[i - 1] * l[i - 1];
            if (i < N - 2){
                u_[i] = U[i];
                p[i] = -l[i - 1] * p[i - 1];
                l[i] = L[i + 1] / d[i];
                r[i] = -u_[i - 1] * r[i - 1] / d[i];
                d[N - 1] -= r[i] * p[i];
            }
            else {
                u_[i] = U[i] - p[i - 1] * l[i - 1];
                l[i] = (L[i + 1] - u_[i - 1] * r[i - 1]) / d[i];
                d[N - 1] -= l[i] * u_[i];
            }
        }

    }
    // time integration step
    void time_int(int t, int type_scheme, int type_linear)
    {
        switch (type_scheme){
            case T_SCHEME::F_EULER:
                explicit_step(dt, u, u, u_new);
                swap();
                break;

            case T_SCHEME::RK2:
                explicit_step(0.5 * dt, u, u, u_new);
                explicit_step(dt, u, u_new, u_new);
                swap();
                break;

            case T_SCHEME::RK4:
                //  k1-term addition
                explicit_step(dt/6.0,   u,      u,      u_new); //u_new = u_n + dt/6*RHS(u_n)
                //  k2-term addition
                explicit_step(0.5*dt,   u,      u,      k_tmp); //u* = u_n + dt/2*RHS(u_n) -> k_tmp
                explicit_step(dt/3.0,   u_new,  k_tmp,  k_tmp2);//u_new = u_new + dt/3*RHS(u*) -> k_tmp2
                //  k3-term addition
                explicit_step(0.5*dt,   u,      k_tmp,  u_new); //u** = u_n + dt/2*RHS(u*) -> u_new
                explicit_step(dt/3.0,   k_tmp2, u_new,  k_tmp); //u_new = u_new + dt/3*RHS(u**) -> k_tmp
                //  k4-term addition
                explicit_step(dt,       u,      u_new,  k_tmp2);// u*** = u_n + dt*RHS(u**) -> k_tmp2
                explicit_step(dt/6.0,   k_tmp,  k_tmp2, u);     //u_n = u_new + dt/6*RHS(u***) -> k_tmp

                break;

            case T_SCHEME::B_EULER:
                implicit_step(t, type_linear);
                break;
            case T_SCHEME::CN:
                implicit_step(t, type_linear, true);
                break;

        }
    }

    void solve(string file, int type_scheme = T_SCHEME::F_EULER, int type_linear = T_LINEAR::FINITE_DIFF_DU_DT, int n_iter_write = 1, bool debug = true)
    {
        initial_conditions();
        if (type_scheme == T_SCHEME::RK4){
            k_tmp = new double[nx + 2 * ng];
            k_tmp2 = new double[nx + 2 * ng];
        }
        if (type_scheme == T_SCHEME::B_EULER || type_scheme == T_SCHEME::CN || type_scheme == T_SCHEME::CN_NR) {
            L = new double[nx]; D = new double[nx]; U = new double[nx];
            p = new double[nx - 2]; r = new double[nx - 2]; dfL = new double[nx + 2 * ng];
            if (type_scheme == T_SCHEME::CN_NR) {
                du = new double[nx];fL_new = new double[nx + 2 * ng]; q_new = new double[nx + 2 * ng];
            }
        }

        output.open(file);
        write_result();
        for (size_t t = 0; t < max_steps; t++){
            time_int(t, type_scheme, type_linear);
            if (t % n_iter_write == 0)
                write_result();
            if (debug)print_result();
        }
        output.close();
        if (type_scheme == T_SCHEME::RK4) {
            delete [] k_tmp;
            delete [] k_tmp2;
        }
        if (type_scheme == T_SCHEME::B_EULER || type_scheme == T_SCHEME::CN || type_scheme == T_SCHEME::CN_NR) {
            delete[] L; delete[] D; delete [] U;
            delete[] p; delete[] r; delete[] dfL;
            if (type_scheme == T_SCHEME::CN_NR) {
                delete[] du; delete[] fL_new; delete[] q_new;
            }
        }
    }

    void write_result()
    {
        for (size_t i = ng; i < nx + ng; i++){
            output << u[i] << "\t";
        }
        output << u[ng];// periodic bc
        output<< "\n";
    }
    void write_mesh(string file)
    {
        ofstream out(file);
        for (size_t i = ng; i <= nx + ng; i++) {
            out << x[i] << "\t";
        }
        out.close();
    }
    void write_uulr(string file)// u, u_left and u_right
    {
        update_ghost(u);
        // need in reconstruct from last value u
        reconstruct(u, true);

        ofstream out(file);
        // x - for uL
        for (size_t i = ng; i <= nx + ng; i++) {
            out << 0.5*(x[i] + x[i-1]) << "\t";
        }
        out << endl;
        // uL
        for (size_t i = ng; i <= nx + ng; i++) {
            out << uL[i] << "\t";
        }
        out << endl;

        // x for uR
        for (size_t i = ng; i <= nx + ng; i++) {
            out << 0.5 * (x[i] + x[i + 1]) << "\t";
        }
        out << endl;
        // uR
        for (size_t i = ng; i <= nx + ng; i++) {
            out << uR[i] << "\t";
        }
        out << endl;
        // x for u
        for (size_t i = ng; i <= nx + ng; i++) {
            out <<x[i] << "\t";
        }
        out << endl;
        // u
        for (size_t i = ng; i <= nx + ng; i++) {
            out << u[i] << "\t";
        }

        out.close();
    }
    void print_result()
    {
        for (size_t i = ng; i <= nx + ng; i++) {
            cout << u[i] << "\t";
        }
        cout << "\n";
    }

    // free memory
    ~FV_Burgers1D()
    {
        delete[] x;
        delete[] u;
        delete[] u_new;
        delete[] uL;
        delete[] uR;
        delete[] fL;
        delete[] q;
    }
};


int main()
{
    // create sample with defaults settings
    int n = 100;
    double xL = -1, xR = 1, vis = 0.002, dt = 0.001, time = 4.0;
    //int saved_timesteps = 500; int(time / dt / saved_timesteps + 0.5);
    FV_Burgers1D sample(n, xL, xR, vis, dt, time, no_limiters, tophat, rare_in_center);

    //sample.solve(string("sol_CN_NR.txt"), T_SCHEME::CN_NR, T_LINEAR::NO, 1, false);// write each step solution to file


    // Explicit schemes
    // for dt = 0.01 these schemes unstable // for explicit schemes parameters of linearization ignored!
    //sample.solve(string("sol_F_Euler.txt"), T_SCHEME::F_EULER, T_LINEAR::NO, 1, false);// write each step solution to file
    //ample.solve(string("sol_RK4.txt"), T_SCHEME::RK4, T_LINEAR::NO, 1, false);// write each step solution to file

    //
    // Implicit schemes
    //
    // these schemes more stable then explicit
    // this kind of linearization have high order of accuracy
    sample.solve(string("sol_B_Euler_DU_EQ_RHS.txt"), T_SCHEME::F_EULER, T_LINEAR::DU_DT_EQUAL_RHS, 1, false);// write each step solution to file
    //sample.solve(string("sol_CN_DU_EQ_RHS.txt"), T_SCHEME::CN, T_LINEAR::DU_DT_EQUAL_RHS, 1, false);// write each step solution to file

    // this type of linearization has high stability, but less accuracy
    //sample.solve(string("sol_B_Euler_DU_FD.txt"), T_SCHEME::B_EULER, T_LINEAR::FINITE_DIFF_DU_DT, 1, false);// write each step solution to file
    //sample.solve(string("sol_CN_DU_FD.txt"), T_SCHEME::CN, T_LINEAR::FINITE_DIFF_DU_DT, 1, false);// write each step solution to file

    //sample.solve(string("sol_CN_NR.txt"), T_SCHEME::CN_NR, T_LINEAR::NO, 1, false);// write each step solution to file

    sample.write_mesh(string("mesh.txt"));
    sample.print_result();
    //sample.write_uulr(string("u_left_right.txt"));


    return 0;
}
