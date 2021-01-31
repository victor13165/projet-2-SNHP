
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>


#define NthMax 128 //Définit le nombre de threads max/block
#define MAX(a,b) (((a)>(b))?(a):(b))

//fonction fournie
__device__ __forceinline__ double atomMax(double* address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while (val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long*)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}


// Operation de reduction depuis le device
// in : tab double *a
// out : *maxi = max(a)
// algo : arbre descendant entre threads (O(log N)) + atomicMax (double)
// mem : stockage en shared des portions de a
__device__ void redMaxDblOptDev(int N, double* a, double* maxi)
{
    __shared__ double tmp[NthMax];
    int idx = blockIdx.x * blockDim.x + threadIdx.x, s;

    if (idx < N)
    {
        tmp[threadIdx.x] = a[idx];
    }
    else
    {
        tmp[threadIdx.x] = 0.;
    }

    __syncthreads();

    for (s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            tmp[threadIdx.x] = MAX(tmp[threadIdx.x], tmp[threadIdx.x + s]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomMax(maxi, tmp[0]);
    }
}

//A utiliser pour calculer le max du tableau a dans la valeur maxi à partir de code CPU
__global__ void redMaxDblOptKer(int N, double* a, double* maxi)
{
    redMaxDblOptDev(N, a, maxi);
}

// initialisation des champs
void init(int N, double dx, double* x, double* h, double* hu)
{
    int i;
    for (i = 0; i < N; i++)
    {
        x[i] = ((double)i + 0.5) * dx;      // espace
        if (x[i] < 0.5)  h[i] = 10.0;  // hauteur : discontinue en x=0.5
        else          h[i] = 1.0;   //
        hu[i] = 0.0;                // vitesse nulle partout => hu = 0 !

    }
}

/* 
integration connaissant les flux
 f_h et f_hu
Vol. finis : u[i]^(n+1) = u[i]^n - dt( F[i+1/2] - F[i-1/2] )

Paramètres : 
    - N : nombre de points
    - dx : pas de discrétisation
    - cm : célérité max calculée avec la fonction Flux_kernel()
    - h : vecteur taille N de la hauteur à l'instant n
    - hu : vecteur taille N de la hauteur*vitesse à l'instant n
    - fh : vecteur taille N flux hauteur
    - fu : vecteur taille N flux vitesse
*/
__global__ void integre_kernel(int N, double* dx, double* cm, double* h, double* hu, double* fh, double* fu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; //On récupère le thread qui va faire le calcul (seulement dans des blocs de dimension 1
    double dt = 0.75 * (*dx) / (*cm); //On calcule le nouveau pas de temps
    double rap = dt / (*dx); //Constante de calcul

    if (i >= 1 && i <= N - 2) { //si on n'est pas au bord calcul normal
        h[i] = h[i] - rap * (fh[i] - fh[i - 1]);
        hu[i] = hu[i] - rap * (fu[i] - fu[i - 1]);
    }
    else if (i == 0) { //Condition limite en 0
        h[0] = h[1];  hu[0] = hu[1]; // gauche => sortie libre : hg = hd, ug = ud
    }
    else { //Condition limite en N-1
        h[N - 1] = h[N - 2]; hu[N - 1] = -hu[N - 2]; // droite => mur : hg = hd, ug = -ud
    }    
}

//Calcule le flux à partir des hg,hd,ug,ud et calcule la valeur max
/*
Paramètres:
    - hg,hd : hauteur gauche/droite
    - ug,ud : vitesse gauche/droite
    - fh : vecteur taille N flux hauteur
    - fu : vecteur taille N flux vitesse
    - cmax : célérité max
*/
__device__ void calculFLux_kernel(double hg, double hd, double ug, double ud, double* fh, double *fu, double *cmax) {
    double g = 9.81, cm, cg, cd, c1, c2;
    cg = sqrt(g * hg);  cd = sqrt(g * hd);

    // Calcul des vitesse d'onde
    c1 = fmin(ug - cg, ud - cd);
    c2 = fmax(ug + cg, ud + cd);

    if (c1 >= 0.0) { // toutes les ondes traversent par la droite
        *fh = hg * ug;
        *fu = hg * ug * ug + 0.5 * g * hg * hg;
        cm = fabs(c2);
    }
    else if (c2 <= 0.0) { // toutes les ondes traversent a gauche
        *fh = hd * ud;
        *fu = hd * ud * ud + 0.5 * g * hd * hd;
        cm = fabs(c1);
    }
    else {   // cas ou l'on a un probleme de Riemann
        // Flux HLL F = (c2*fg - c1*fd)/(c2-c1)  + c1*c2*(Ud - Ug)/(c2-c1)
        *fh = (c2 * hg * ug - c1 * hd * ud) / (c2 - c1) + c1 * c2 * (hd - hg) / (c2 - c1);
        *fu = (c2 * (hg * ug * ug + 0.5 * g * hg * hg) - c1 * (hd * ud * ud + 0.5 * g * hd * hd)) / (c2 - c1) + c1 * c2 * (hd * ud - hg * ug) / (c2 - c1);
        cm = fmax(fabs(c1), fabs(c2));
    }

    *cmax = fmax(*cmax, cm); // vitesse d'onde max
}

/*Résoudre les flux pour chaque point
* Tableau cm_tot prend toutes le valeurs calculées par chaque thread et on utilise la réduction pour trouver la valeur max stockée dans *cmax
Paramètres:
    - N : nombre de points
    - h : vecteur taille N de la hauteur à l'instant n
    - hu : vecteur taille N de la hauteur*vitesse à l'instant n
    - fh : vecteur taille N flux hauteur
    - fu : vecteur taille N flux vitesse
    - cm_tot_d : vecteur taille N qui stocke la célérité max calculée par chaque thread i
    - cmax : célérité max sur laquelle on va faire la réduction du vecteur cm_tot_d
*/
__global__ void Flux_kernel(int N, double* h, double* hu, double* fh, double* fu, double* cm_tot_d, double* cmax)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; //On récupère le thread qui va faire le calcul
    
    double g = 9.81, cm, cg, cd, c1, c2;
    double hg, hd, ug, ud;

    if (i <= N - 2) { //Si on est l'avant dernier thread: on fait le calcul de flux

        hg = h[i]; hd = h[i + 1];
        ug = hu[i] / hg; ud = hu[i + 1] / hd;
        calculFLux_kernel(hg, hd, ug, ud, &fh[i], &fu[i], &cm_tot_d[i]); //Calcul du flux et stockage de la vitesse max dans le tableau des vitesses max
    }

    redMaxDblOptDev(N, cm_tot_d, cmax); //Réduction dans cmax
}

// ecriture des résultats dans un fichier
// sauvegarde de h et u
void ecrit(int N, FILE* fichier, double* x, double* h, double* hu)
{
    int i;
    for (i = 0; i < N; i++)
    {
        fprintf(fichier, "%lf %lf %lf\n", x[i], h[i], hu[i] / h[i]);
    }
    fclose(fichier); // fermeture
}

//fonction principale
//Nb de threads par block maxi : 1024
int main(int argc, char* argv[])
{
    int N, Nt; //Nombre de points discrétisation, itérations et nombre maxi de threads/bloc
    int it; //Pour l'itération en temps
    double dx; //Variables CPU
    double *dx_d, *cm_d; //Variables GPU, 

    //Tableaux CPU
    double* x, * h, * hu;

    //Tableaux GPU
    double* x_d, * h_d, * hu_d, * fh_d, * fu_d, * cm_tot_d;

    FILE* fparam, *finit, * fres; //Fichiers de sauvegarde

    fparam = fopen("param.dat", "r");
    fscanf(fparam, "%d %d\n", &N, &Nt);
    fclose(fparam);

    //Dimensions grille/block
    dim3 grid(N/NthMax + 1, 1, 1); 
    dim3 block(NthMax, 1, 1);

    printf("N : %d , Nt : %d\n", N, Nt); //Check paramètres de calcul 
    printf("Nblocks : %d , Nthreads/block : %d\n", N / NthMax + 1, NthMax); //Check répartition threads/blocks

    // ouverture des fichiers
    finit = fopen("initial.dat", "w"); 
    fres = fopen("finale.dat", "w");

    dx = 1.0 / (double)(N); //Pas de discrétisation spatiale

    //Allocation CPU, pas de fu ou fh car on va faire tous les calculs dans le GPU
    x =  (double*)malloc(N * sizeof(double));  //Abcisse
    h =  (double*)malloc(N * sizeof(double)); //Hauteur
    hu = (double*)malloc(N * sizeof(double)); //Hauteur*vitesse


    //Allocation GPU : suffixe "_d" pour "device"
    cudaMalloc((void**) &h_d, N * sizeof(double)); 
    cudaMalloc((void**)&hu_d, N * sizeof(double)); 
    cudaMalloc((void**)&fh_d, N * sizeof(double));
    cudaMalloc((void**)&fu_d, N * sizeof(double)); 
    cudaMalloc((void**)&cm_tot_d, N * sizeof(double)); //va stocker toutes les valeurs des célérités max de chaque thread
    cudaMalloc((void**)&cm_d, sizeof(double)); //Valeur max sur tout le domaine sur laquelle on va effectuer la réduction de cm_tot_d
    cudaMalloc((void**)&dx_d, sizeof(double)); //pas d'espace

    init(N, dx, x, h, hu); //Initialisation des champs h et hu 
    ecrit(N, finit, x, h, hu); //Sauvegarde dans le fichier "initial.dat"

    //Copie des variables que l'on a initialisées sur le host sur les variables device
    cudaMemcpy(h_d, h, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(hu_d, hu, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dx_d, &dx,sizeof(double), cudaMemcpyHostToDevice);

    for (it = 1; it <= Nt; it++)   // boucle en temps
    {
        Flux_kernel <<<grid, block>>> (N, h_d, hu_d, fh_d, fu_d, cm_tot_d, cm_d);     // calcul des flux HLL et vitesse d'onde (cm)
        integre_kernel <<<grid, block>>> (N, dx_d, cm_d, h_d, hu_d, fh_d, fu_d); // integration sur un pas de temps Vol. Finis
    }

    //Récupérer les valeurs de h et hu calculées
    cudaMemcpy(h, h_d, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hu, hu_d, N * sizeof(double), cudaMemcpyDeviceToHost);

    ecrit(N, fres, x, h, hu); //Ecrire résultats

    free(x); free(h); free(hu);
    cudaFree(h_d); cudaFree(hu_d); cudaFree(fh_d); cudaFree(fu_d); cudaFree(cm_tot_d); cudaFree(cm_d); cudaFree(dx_d);
    return 0;
}

