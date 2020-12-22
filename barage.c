#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// initialisation des champs
void init(int N, double dx, double *x, double *h, double *hu)
{
  int i;
  for(i=0;i<N;i++)
  {
    x[i] = ((double)i + 0.5)*dx;      // espace
    if(x[i]<0.5)  h[i] = 10.0;  // hauteur : discontinue en x=0.5
    else          h[i] = 1.0;   //
    hu[i] = 0.0;                // vitesse nulle partout => hu = 0 !

  }
}
// integration connaissant les flux
// f_h et f_hu
// Vol. finis : u[i]^(n+1) = u[i]^n - dt( F[i+1/2] - F[i-1/2] )
void integre(int N,double dt,double dx, double *h, double *hu, double *fh, double *fu)
{
  int i;
  double rap = dt/dx;

  for(i=1;i<=N-2;i++)
  {
    h[i] = h[i]  - rap*(fh[i]-fh[i-1]);
    hu[i]= hu[i] - rap*(fu[i]-fu[i-1]);
  }
  // conditions aux limites
  h[0] = h[1];  hu[0] = hu[1]; // gauche => sortie libre : hg = hd, ug = ud
  h[N-1] = h[N-2]; hu[N-1] = -hu[N-2]; // droite => mur : hg = hd, ug = -ud
}


// Calcul des flux
// Schema HLL : flux solution au travers des surface
// => resolution d'un probleme de riemann au travers de chaque surface
void calculFlux(double hg, double hd, double ug, double ud, double *fh, double *fu, double *cmax)
{
  double g = 9.81, cm,cg,cd,c1,c2;

  cg = sqrt(g*hg);  cd = sqrt(g*hd);
  // Calcul des vitesse d'onde
  c1 = fmin(ug - cg , ud - cd);
  c2 = fmax(ug + cg , ud + cd);

  if(c1 >= 0.0) { // toutes les ondes traversent par la droite
    *fh = hg*ug ;
    *fu = hg*ug*ug + 0.5*g*hg*hg;
    cm = fabs(c2);
  } else if (c2 <= 0.0) { // toutes les ondes traversent a gauche
    *fh = hd*ud ;
    *fu = hd*ud*ud + 0.5*g*hd*hd;
    cm = fabs(c1);
  } else {   // cas ou l'on a un probleme de Riemann
    // Flux HLL F = (c2*fg - c1*fd)/(c2-c1)  + c1*c2*(Ud - Ug)/(c2-c1)
    *fh = ( c2*hg*ug - c1*hd*ud )/(c2-c1) + c1*c2*(hd-hg)/(c2-c1);
    *fu= ( c2*( hg*ug*ug + 0.5*g*hg*hg ) - c1*( hd*ud*ud + 0.5*g*hd*hd ))/(c2-c1) + c1*c2*(hd*ud-hg*ug)/(c2-c1);
    cm = fmax(fabs(c1),fabs(c2));
  }

  *cmax = fmax(*cmax,cm); // vitesse d'onde max

}

//Résoudre les flux pour chaque point
double Flux(int N, double *h, double *hu, double *fh, double *fu)
{
  int i;
  double g=9.81,cm,cmax=0.0;
  double hg, hd, ug, ud;
  for(i=0;i<=N-2;i++)
  {
    hg =  h[i]   ; hd =  h[i+1];
    ug = hu[i]/hg; ud = hu[i+1]/hd;
    calculFlux(hg,hd,ug,ud,&fh[i],&fu[i],&cmax);
  }

  return cmax;
}
// ecriture des résultats dans un fichier
// sauvegarde de h et u
void ecrit(int N,FILE* fichier,double *x, double *h, double *hu)
{
  int i;
  for(i=0;i<N;i++)
  {
    fprintf(fichier,"%lf %lf %lf\n",x[i],h[i], hu[i]/h[i]);
  }
  fclose(fichier); // fermeture
}

void afficheTab(int N, double *tab)
{
  int i;

  for (i = 0; i < N; i++)
  {
    printf("%lf\n",tab[i]);
  }
  printf("\n");
}

//fonction principale
int main(int argc, char* argv[])
{
  int N=100,Nt=100, it;
  double dx,dt,cm;
  double *x,*h, *hu, *fh, *fu;
  FILE *finit, *fres;

  finit = fopen("initial.dat","w"); // ouverture des fichiers
  fres  = fopen("finale.dat","w");

  printf("%d %d\n",N,Nt);
  dx = 1.0/(double) (N);
  dt = 0.00000001;

  x  = malloc(N*sizeof(double));  // allocation sur N points
  h  = malloc(N*sizeof(double));
  hu = malloc(N*sizeof(double));
  fh = malloc(N*sizeof(double));
  fu = malloc(N*sizeof(double));

  init(N,dx,x,h,hu);       // initialisation et sauvegarde
  ecrit(N,finit,x,h,hu);

  for(it=1;it<=Nt;it++)   // boucle en temps
  {
    cm = Flux(N,h,hu,fh,fu);     // calcul des flux HLL et vitesse d'onde (cm)
    // afficheTab(N,fh);
    // dt = 0.75*dx/cm;              // calcul du pas de temps admissible
    // integre(N,dt,dx,h,hu,fh,fu); // integration sur un pas de temps Vol. Finis
  }

  ecrit(N,fres,x,fh,fu);
  free(x); free(h); free(hu); free(fh); free(fu);
  return 0;
}
