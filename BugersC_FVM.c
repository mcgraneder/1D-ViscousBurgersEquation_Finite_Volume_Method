#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <stdbool.h>
#define PI 3.14
#define max(a,b)             \
({                           \
    __typeof__ (a) _a = (a); \
    __typeof__ (b) _b = (b); \
    _a > _b ? _a : _b;       \
})

//slope limiter function prototypes
double no_limiters(double q1, double q2);
double minmod(double q1, double q2);
double superbee(double q1, double q2);
double vanAlbada(double q1, double q2);
double vanLeer(double q1, double q2);

//initialization function prototypes
void init_mesh(double* x, double min_x, double max_x, int nx, int ng, bool is_structured);
void initial_conditions(double* u, double* u_new, double* x, double min_x, double max_x, double nx, double ng );
void update_ghost(double *u, int nx, int ng);
void reconstruct(double *u, double* q, double* uL, double* uR, double* fL, int nx, int ng, double (*phi)(double, double), bool only_uLR, bool is_implicit1);

//explicit scheme function prototypes
void RHS(double *u, double *u_new, double* x, double* q, double* fL, double nu, int nx, int ng);
void explicit_step(double dt, double *u, double* u_old, double *u_new, int nx, int ng);
void copy_solution(double* u, double* u_new, int nx, int ng);

//implicit scheme function protopypes
void implicit_step(double* u, double* u_new, double* x, double* dfL, double* fL, double* uL, double* q, double* L, double* D, double* U, double* p, double* r, double dt, double nu, int nx, int ng, bool is_CN);
void LU_backward_steps(double* u_new, double* L, double* D, double* U, double* p, double* r, size_t nx, int ng, double* u);
void LU_sparse(double *L, double *D, double *U, double *p, double *r, size_t nx, double* u);//general functions

void writeSolutionToFile(double* u, int ng, int nx, bool is_final_timestep);
void writeSolutionToFileImplicit(double* u, int ng, int nx, bool is_final_timestep);
void burgers_analytical_solution(double* u_analytical, double* x, double t1, double nu, int nx, int ng, FILE* analytical_fp);



//main function
int main()
{

    //conditional statmentents
    FILE* analytical_fp = NULL;
    analytical_fp = fopen("BurgersAnalytical.txt", "w");

    int type_scheme =  1;
    bool only_uLR = false;
    bool is_implicit1 = false;

    //pointer function for slope limiterd
    double (*phi)(double, double);

    int max_steps = 500;
    int nx = 100;
    int ng = 2;
    //double min_x = 0;
    //double max_x = 2 * PI;
    //double nu = 0.02;
    double min_x = -1;
    double max_x = 1;
    double nu = 0.0025;
    double stop_time = 4.0;
    double dt = stop_time / max_steps;
    double t1;

    //arrays for numerical scheme
    double* x = (double*)calloc(nx + 2 * ng, sizeof(double));
    double* u = (double*)calloc(nx + 2 * ng, sizeof(double));
    double* u_analytical = (double*)calloc(nx + 2 * ng, sizeof(double));
    double* u_old = (double*)calloc(nx + 2 * ng, sizeof(double));
    double* u_new = (double*)calloc(nx + 2 * ng, sizeof(double));
    double* uL = (double*)calloc(nx + 2 * ng, sizeof(double));
    double* uR = (double*)calloc(nx + 2 * ng, sizeof(double));
    double* fL = (double*)calloc(nx + 2 * ng, sizeof(double));
    double* q = (double*)calloc(nx + 2 * ng, sizeof(double));

    //arrays for implicit scheme and sparse matrix solver
    double* U = (double*)calloc(nx + 2 * ng, sizeof(double));
    double* L = (double*)calloc(nx + 2 * ng, sizeof(double));
    double* D = (double*)calloc(nx + 2 * ng, sizeof(double));
    double* p = (double*)calloc(nx + 2 * ng, sizeof(double));
    double* r = (double*)calloc(nx + 2 * ng, sizeof(double));
    double* dfL = (double*)calloc(nx + 2 * ng, sizeof(double));


   init_mesh(x, min_x, max_x, nx, ng, true);
   initial_conditions(u, u_new, x, min_x, max_x, nx, ng );

   //begin time iteration loop and solve seqatio. Two cases the firstly
   //the first case solves the burgers equation explicitly and the second
   //case solves the burgers equation via implict methods. in the functions
   // 'implicit_step' change the bool from true to false to chage between
   //crank nichilson and backward euler schemes
   switch (type_scheme)
   {
        case 1:
            for (int i = 0; i < max_steps; i ++)
           {
               t1 = i * dt;
                   //firstly we want to solve the burgers equation via the analtytical for max timesteps
               burgers_analytical_solution(u_analytical, x, t1, nu, nx, ng, analytical_fp);

               //printf("\n\n\n");
               update_ghost(u, nx, ng);
               reconstruct(u, q, uL, uR, fL, nx, ng, no_limiters, false, false );
               RHS(u, u_new, x, q, fL, nu, nx, ng);
               explicit_step(dt, u, u, u_new, nx, ng);


               copy_solution(u, u_new, nx, ng);

               //write each timestep solution to a new file
               if (i % 10 == 0)
               {
                   writeSolutionToFile(u, ng, nx, false);
               }

           }

           //print final timestep to seperate file
           writeSolutionToFile(u, ng, nx, true);

           break;

        case 2:

             for (int i = 0; i < max_steps; i ++)
               {
                   //printf("\n\n\n");
                   update_ghost(u, nx, ng);
                   reconstruct(u, q, uL, uR, fL, nx, ng, minmod, false, true );
                   implicit_step(u, u_new, x, dfL, fL, uL, q, L, D, U, p, r, dt, nu, nx, ng, true);

                   if (i % 10 == 0)
                   {
                       writeSolutionToFileImplicit(u, ng, nx, false);
                   }

                }
                writeSolutionToFileImplicit(u, ng, nx, true);
               

            break;
   }


   //print final timestep
   printf("\n");
   for (int i = ng; i < nx + ng; i++)
   {
       printf("%f\n", u_new[i]);
   }



    return EXIT_SUCCESS;
}


//program functions
//function that generates computational mesh. (unifrom)
//TO DO -- add unstructured mesh
void init_mesh(double* x, double min_x, double max_x, int nx, int ng, bool is_structured)
{

    double L = abs(max_x - min_x);
    for (size_t i = 0; i <= nx; i++)
    {
        x[i + ng] = min_x + L * ((double)(i) / nx);
    }

    // extrapolate to ghost nodes
    for (size_t i = 0; i < ng; i++) {
        x[i] = x[i + nx] - L;
        x[i + ng + nx] = x[i + ng] + L;

    }
    for (int i = 0; i <= nx + 2 * ng; i++)
    {
        //cout << x[i] << "\n";
        //printf("%f\n", x[i]);
    }
}


//function that generates initial conditions. Cuurenly have square wave function and the burgers analytical solution
//as the inital condition either
//TO DO-- add sine wave and heavyside initial conditions
void initial_conditions(double* u, double* u_new, double* x, double min_x, double max_x, double nx, double ng  )
{
    double center = 0.5 * (min_x + max_x);
    double _dx = 0.5 * (max_x - min_x) / 4.0;

    double* t = (double*)calloc(nx + 2 * ng, sizeof(double));
    double NU = 0.07;
    double phi;
    double dphi;

    int cas = 1;
    switch(cas)
    {
        case 1:

            for (int i = 0; i < nx + 2*ng; i++)
            {
                if (x[i] < center - _dx || x[i] > center + _dx)
                {
                    u[i] = 0.0;
                    u_new[i] = 0.0;
                }

                else{
                    u[i] = 1.0;
                    u_new[i] = 1.0;

                }
                //printf("%f, %f\n", x[i], u[i]);
            }
            break;

        case 2:

            for (int i = 0; i < nx + 2 * ng; i++)
            {
                //get rid of ts
                phi = exp(-pow((x[i] - 4 * t[i]), 2) / (4 * NU * (t[i] + 1)))
                    + exp(-pow((x[i] - 4 * t[i] - 2 * PI), 2) / (4 * NU * (t[i] + 1)));

                dphi = (-0.5 * (x[i] - 4 * t[i]) / (NU * (t[i] + 1)) * exp(-pow((x[i] - 4 * t[i]), 2) / (4 * NU * (t[i] + 1)))
                    - 0.5 * (x[i] - 4 * t[i] - 2 * PI) / (NU * (t[i] + 1)) * exp(-pow((x[i] - 4 * t[i] - 2 * PI), 2) / (4 * NU * (t[i] + 1))));

                u[i] = -2.0 * NU * dphi / phi + 4.0;
                u_new[i] = -2.0 * NU * dphi / phi + 4.0;
                //un[i] = u[i];

                //printf("%f\n", u[i]);
                //fprintf(initial_fp1, "%f\n", u[i]);
            }
            break;

          case 3:

              for (int i = 0; i < nx + 2 * ng; i++)
              {
                u[i] = 0.5 + 0.5 * sin(1*(2.*(x[i] - (-1))/(1 - (-1))-1.0)*PI);

                printf("%f\n", u[i]);
              }
    }
}

//function that creates periodicity by copying the first two ghost cells to the last two internal cells and
// the last two ghost cells to the first two interna cells at the beginning of each time iteration,
void update_ghost(double *u, int nx, int ng)
    {
        //int ng = 2;
        for (size_t i = 0; i < ng; i++) {
            u[i] = u[i + nx];
            u[i + ng + nx] = u[i + ng];
        }
    }


//function that reconstructs the value of u at the left and righ boundary of the cell and from this calculates the convecive flux
//note if we are solving via implicit schemes the derivative of the flux is also calculated (burgers flux = 0.5 * u^2)
void reconstruct(double *u, double* q, double* uL, double* uR, double* fL, int nx, int ng, double (*phi)(double, double), bool only_uLR, bool is_implicit1)
    {

        for (size_t i = ng-1; i <= nx + ng + 1; i++)
        {
            q[i] = u[i] - u[i - 1];
        }

        for (size_t i = ng-1; i <= nx + ng; i++)
        {
            //double du = 0.5 * phi(q[i], q[i + 1]) * q[i + 1];
            uL[i] = u[i];
            uR[i] = u[i];
        }

        if (! only_uLR)
        {
            for (size_t i = ng; i <= nx + ng; i++)
            {
                fL[i] = 0.5 * (pow(uR[i-1], 2) / 2 + pow(uL[i], 2) / 2 - (abs(uL[i - 1]))*(uL[i] - uR[i-1]) );
                //printf("%f\n", fL[i] );
            }
        }

        // condition for implicit schemes
        double* dfL = (double*)calloc(nx + 2 * ng, sizeof(double));

        if (is_implicit1) {
                for (size_t i = ng; i <= nx + ng; i++) {
                    dfL[i] = 0.5 * (uR[i - 1] + uL[i] - max(abs(1), abs(1)) * (uL[i] - uR[i - 1]));
                }
            }
    }


//function that copys the solution form the u_new array back to the u array so that u can be updated for each timestep
void copy_solution(double* u, double* u_new, int nx, int ng)
    {
        for (int i = 0; i < nx + 2 * ng; i++ )
        {
            u[i] = u_new[i];
        }
    }


    //function that calculates the diffusion term on the RHS of the burgers equation (calculates the diffusive flux below)
void RHS(double *u, double *u_new, double* x, double* q, double* fL, double nu, int nx, int ng)
    {
        //update_ghost(u, nx, ng);
        //reconstruct(u, q, uL, uR, fL, nx, ng );

        for (size_t i = ng; i < nx + ng; i++) {
            double dx0 = x[i] - x[i - 1];
            double dx1 = x[i + 1] - x[i];
            double dxi = (dx0 + dx1) / 2.0;
            u_new[i] = - (fL[i + 1] - fL[i] - nu * (q[i + 1] / dx1 - q[i] / dx0)) / dxi;
            //printf("%f\n", u_new[i]);
        }
    }


//explicit time integration funcion function that currently supporst first order forward euler upwinding)
//TO DO -- add rk2, rk3, rk4
void explicit_step(double dt, double *u, double* u_old, double *u_new, int nx, int ng)
    {
        //RHS(u, u_new);
        double u_sum = 0;
        for (size_t i = ng; i < nx + ng; i++)
        {
            u_new[i] = u_old[i] + dt * u_new[i];
            u_sum += u_new[i];

            printf("%f\n", u[i]);
        }
        printf("\n\n");
        printf("%f\n", u_sum);
    }


//implicit time integration function that supports first order backward euler downwinding and 2nd order crank nichilson
 void implicit_step(double* u, double* u_new, double* x, double* dfL, double* fL, double* uL, double* q, double* L, double* D, double* U, double* p, double* r, double dt, double nu, int nx, int ng, bool is_CN)
    {
        double u_sum = 0;

        for (size_t i = ng; i < ng + nx; i++) {
            double dx0 = x[i] - x[i - 1];
            double dx1 = x[i + 1] - x[i];
            double dxi = (dx0 + dx1) / 2.0;
            double dtx = dt / dxi, nu0 = nu / dx0, nu1 = nu / dx1;
            if (is_CN)dtx *= 0.5;
                    
            L[i - ng] = -dtx * (dfL[i] + nu0);
            D[i - ng] = 1.0 + dtx* (dfL[i + 1] - dfL[i] + nu0 + nu1);
            U[i - ng] = dtx * (dfL[i + 1] - nu1);
                    // used u_new for vector rhs=b
            u_new[i - ng] = u[i] - dtx * (fL[i + 1] - dfL[i + 1] * uL[i + 1] - (fL[i] - dfL[i] * uL[i]));
            u_sum += u_new[i + ng];
       
        
                    // additional term for Crank-Nicolson scheme
            if (is_CN)
            {
                u_new[i - ng] -= dtx * (fL[i + 1] - fL[i] - nu * (q[i + 1] / dx1 - q[i] / dx0));
            }
        }

        LU_sparse(L, D, U, p, r, nx, u);

        LU_backward_steps(u_new, L, D, U, p, r, nx, ng, u);
        printf("\n\n%f\n", u_sum);

    }


//baxkwards step function
void LU_backward_steps(double* u_new, double* L, double* D, double* U, double* p, double* r, size_t nx, int ng, double* u)
    {

        // use u_new -> y
        double u_sum = 0;
        u_new[0] = u_new[0];
        u_new[nx - 1] = u_new[nx - 1] - r[0] * u_new[0];
        for (size_t i = 1; i <= nx-2; i++){
            u_new[i] = u_new[i] - L[i - 1] * u_new[i - 1];
            if (i != nx - 2)u_new[nx - 1] -= r[i] * u_new[i];
            else u_new[nx - 1] -= L[nx - 2] * u_new[nx - 2];
        }

       
        u[ng + nx - 1] = u_new[nx - 1] / D[nx - 1];
        u[ng + nx - 2] = (u_new[nx - 2] - u[ng + nx - 1] * U[nx - 2]) / D[nx - 2];
        for (int i = nx-3; i >= 0; i--){
            u[ng + i] = (u_new[i] - u[ng + i + 1] * U[i] - u[ng + nx - 1] * p[i]) / D[i];
            u_sum += u[i + ng];
        }
        printf("\n\n%f\n", u_sum);
    }

    
    void LU_sparse(double *L, double *D, double *U, double *p, double *r, size_t nx, double* u)
    {

        D[0] = D[0]; u[0] = U[0]; p[0] = L[0];
        L[0] = L[1] / D[0]; r[0] = U[nx - 1] / D[0];
        D[nx - 1] = D[nx - 1] - r[0] * p[0];
        for (size_t i = 1; i <= nx-2; i++){
            D[i] = D[i] - u[i - 1] * L[i - 1];
            if (i < nx - 2){
                u[i] = U[i];
                p[i] = -L[i - 1] * p[i - 1];
                L[i] = L[i + 1] / D[i];
                r[i] = -u[i - 1] * r[i - 1] / D[i];
                D[nx - 1] -= r[i] * p[i];
            }
            else {
                u[i] = U[i] - p[i - 1] * L[i - 1];
                L[i] = (L[i + 1] - u[i - 1] * r[i - 1]) / D[i];
                D[nx - 1] -= L[i] * u[i];
            }
        }

    }

//function to compute brgers analytical solutio ove time
void burgers_analytical_solution(double* u_analytical, double* x, double t1, double nu, int nx, int ng, FILE* analytical_fp)
{
    double phi;
    double dphi;

    printf("\n\n\n");
    for (int i = 0; i < nx + 2 * ng; i++)
    {
                            //arrays wont work for t?
        phi = exp(-pow((x[i] - 4 * t1), 2) / (4 * nu * (t1 + 1)))
            + exp(-pow((x[i] - 4 * t1 - 2 * PI), 2) / (4 * nu * (t1 + 1)));

        dphi = (-0.5 * (x[i] - 4 * t1) / (nu * (t1 + 1)) * exp(-pow((x[i] - 4 * t1), 2) / (4 * nu * (t1 + 1)))  
             - 0.5 * (x[i] - 4 * t1 - 2 * PI) / (nu * (t1 + 1)) * exp(-pow((x[i] - 4 * t1 - 2 * PI), 2) / (4 * nu * (t1 + 1))));

        u_analytical[i] = -2.0 * nu * (dphi / phi) + 4.0;
                        //memcpy(uan1, u_analytical, NX * sizeof(double));

        //printf("%f\n", u_analytical[i]);
        //fprintf(u_analytical, "%f\n", u_analytical[i]);
    }

}


//slope limiter functions
//function that execited no slope limiting. returns 0 and thus has no effect on our reconstruction of variables function
double no_limiters(double q1, double q2)
{
    return 0.0;
}


//minmod slope limiter function
double minmod(double q1, double q2)
{
    // r = q[i] / q[i + 1], q[i] = u[i]-u[i-1]
    if (q2 != 0.0)return fmax(0.0, fmin(1.0, q1 / q2));
    if (q1 != 0.0)return 1.0;
    return 0.0;
}

//superbee slope limiter function
double superbee(double q1, double q2)
{
    if (q2 != 0.0) {
        double r = q1 / q2;
        return fmax(0.0, fmax(fmin(1.0, 2.0*r), fmin(r, 2.0)));
    }
    if (q1 != 0.0)return 2.0;
    return 0.0;
}

//van albada slope limiter function
double vanAlbada(double q1, double q2)
{
    if (q2 != 0.0) {
        double r = q1 / q2;
        return (r*r + r) / (r*r + 1.0);
    }
    if (q1 != 0.0)return 1.0;
    return 0.0;
}

//van leer slope limiter function
double vanLeer(double q1, double q2)
{
    if (q2 != 0.0) {
        double r = q1 / q2;
        return (r + abs(r)) / (abs(r) + 1.0);
    }
    if (q1 != 0.0)return 2.0;
    return 0.0;
}


//function that writes eaxh timestep to a new solution file for plotting
void writeSolutionToFile(double* u, int ng, int nx, bool is_final_timestep)
 {
    //printf("\nWriting to file... ");
    //initialise seperate file for final timestep solution
    FILE* final_fp = NULL;
    final_fp = fopen("./out/burgers_final_step_sol.txt", "w");

    static int fileIndex = 0;
    char fileName[100];



    sprintf(fileName, "./out/burgers_sol_%d.txt", fileIndex);
    FILE* file = fopen(fileName, "w");


    // Write cell data
    int i;

    for (i = ng; i < nx + ng; i++)
    {
        if (is_final_timestep)
        {
            fprintf(final_fp, "%f\n", u[i]);
        }
        fprintf(file, "%f\n", u[i]);

    }

    fclose(file);

    fileIndex++;
    //printf("done.\n");
}

void writeSolutionToFileImplicit(double* u, int ng, int nx, bool is_final_timestep)
 {
    //printf("\nWriting to file... ");
    //initialise seperate file for final timestep solution
    FILE* final_fp = NULL;
    final_fp = fopen("./out/burgers_final_step_sol.txt", "w");

    static int fileIndex = 0;
    char fileName[100];



    sprintf(fileName, "./out/burgers_implicit_sol_%d.txt", fileIndex);
    FILE* file = fopen(fileName, "w");


    // Write cell data
    int i;

    for (i = ng; i < nx + ng; i++)
    {
        if (is_final_timestep)
        {
            fprintf(final_fp, "%f\n", u[i]);
        }
        fprintf(file, "%f\n", u[i]);

    }

    fclose(file);

    fileIndex++;
    //printf("done.\n");
}
