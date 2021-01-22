#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// Découpe le domaine pour chaque CPU. Fonction réutilisée à partir du TP fait en cours
// Si le domaine n'est pas divisible en parts égales, le reste de la division est redistribué
// sur chaque sous domaine. Le reste est forcément inférieur au nb total de CPU.
int decoupe(int iCPU, int NCPU, int N, double dx, double *x0)
{
  int NP,reste;

  NP = N / NCPU; // nombre de point par CPU
  reste = N % NCPU; //Reste de la division

  if(iCPU < reste)  // Tant qu'il reste du reste à redistribuer
  {
    NP +=1; // Redistribuer le reste
    *x0 = (double)(NP*iCPU)*dx; // On définit le point de départ pour l'initialisation pour chaque CPU
  }
  else
  { // sinon on n'ajoute rien et on définit le point de départ
    *x0 = (double) (NP*iCPU + reste )*dx;
  }

  return NP;
}


// initialisation des champs à partir du x0 propre à chaque CPU
void init(int N, double dx, double *x, double *h, double *hu, double x0)
{
  int i;
  for(i=0;i<N;i++)
  {
    x[i] = x0 + ((double)i + 0.5)*dx;      // espace
    if(x[i]<0.5)  h[i] = 10.0;  // hauteur : discontinue en x=0.5
    else          h[i] = 1.0;   //
    hu[i] = 0.0;                // vitesse nulle partout => hu = 0 !
  }
}

// integration connaissant les flux
// f_h et f_hu
// Vol. finis : u[i]^(n+1) = u[i]^n - dt( F[i+1/2] - F[i-1/2] )
void integre(int N, int iCPU, int nCPU, double dt, double dx, double *h, double *hu, double *fh, double *fu)
{
  int i, erreur;
  double rap = dt/dx, fum, fhm;
  double a_envoyer[2], a_recevoir[2];
  MPI_Status statut;

  //Résolution standard
  for(i=1;i<=N-2;i++)
  {
    h[i]  = h[i]  - rap*(fh[i]-fh[i-1]);
    hu[i] = hu[i] - rap*(fu[i]-fu[i-1]);
  }

  //CONDITIONS LIMITES
  // --> Pour un CPU quelconque : Il faut que chaque cpu récupère les valeurs de fh[N-1] et fu[N-1] du CPU précédent
  // --> Pour les CPUs 0 et N-1 : On applique les conditions limites

  //On utilise la technique pair/impair pour les envois et réception vu en cours
  if (iCPU%2==0) //Cas pair : on envoie en premier
  {

    if (iCPU != nCPU-1) //Tout le monde doit envoyer au CPU+1 sauf CPU N-1
    {
      a_envoyer[0] = fh[N-1]; a_envoyer[1] = fu[N-1]; // Les données à envoyer : les dernières valeurs du CPU courant
      // On envoie au CPU + 1
      erreur = MPI_Send(a_envoyer, 2 , MPI_DOUBLE_PRECISION, iCPU+1  , 257 + iCPU, MPI_COMM_WORLD);

    }

    if (iCPU != 0) //Tout le monde doit recevoir du CPU-1 sauf CPU 0
    {
      //On reçoit du CPU - 1
      erreur = MPI_Recv(a_recevoir,2,MPI_DOUBLE_PRECISION, iCPU - 1, 257 + iCPU - 1, MPI_COMM_WORLD, &statut);
      fhm = a_recevoir[0]; fum = a_recevoir[1]; //Données stockées pour calculer h[0] et hu[0]
    }

  }
  else //Cas impair : on reçoit en premier
  {
    //On reçoit du CPU - 1
    erreur = MPI_Recv(a_recevoir,2,MPI_DOUBLE_PRECISION, iCPU - 1, 257 + iCPU - 1, MPI_COMM_WORLD, &statut);
    fhm = a_recevoir[0]; fum = a_recevoir[1]; //Données stockées pour calculer h[0] et hu[0]

    if (iCPU != nCPU-1) //Tout le monde doit envoyer sauf CPU N-1
    {
      a_envoyer[0] = fh[N-1]; a_envoyer[1] = fu[N-1]; // Les données à envoyer : la fin du domaine du CPU courant
      // On envoie au CPU + 1
      erreur = MPI_Send(a_envoyer, 2 , MPI_DOUBLE_PRECISION, iCPU+1  , 257 + iCPU, MPI_COMM_WORLD);
    }

  }

  /*
  Très vicieux la synthaxe h[0] = h[1]; hu[0] = hu[1]... ça m'a valu
  plusieurs jours de débugage...
  */
  if (iCPU != 0) //Si on n'est pas 0, on résout le point 0 avec les éléments envoyés précédemment
  {
    h[0] = h[0] - rap*(fh[0] - fhm);
    hu[0] = hu[0] - rap*(fu[0] - fum);
  }
  else //Sinon on résout le point 0 avec la condition limite en 0
  {
    h[0] = h[1];
    hu[0] = hu[1];
  }

  if (iCPU != nCPU - 1) //Si on n'est pas N-1, on résout le point N-1 normalement
  {
    h[N-1] = h[N-1] - rap*(fh[N-1] - fh[N-2]);
    hu[N-1] = hu[N-1] - rap*(fu[N-1] - fu[N-2]);
  }
  else //Sinon, on résout le point N-1 avec la condition limite en N-1
  {
    h[N-1] = h[N-2];
    hu[N-1] = -hu[N-2];
  }

}

// Calcul des flux
// Schema HLL : flux solution au travers des surface
// => resolution d'un probleme de riemann au travers de chaque surface
double calculFlux(const double hg, const double hd, const double ug, const double ud, double *fh, double *fu, double *cmax)
{
  double g = 9.81, cm, cg, cd, c1, c2;

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
    *fh = ( c2*hg*ug - c1*hd*ud )/(c2-c1) + c1*c2*(hd-hg)/(c2-c1);
    *fu = ( c2*( hg*ug*ug + 0.5*g*hg*hg ) - c1*( hd*ud*ud + 0.5*g*hd*hd ))/(c2-c1) + c1*c2*(hd*ud-hg*ug)/(c2-c1);
    cm = fmax(fabs(c1),fabs(c2));
  }

  *cmax = fmax(*cmax,cm); // vitesse d'onde max
}


//Calcul de flux pour chaque point du domaine
// PARALLELISATION OK
double Flux(int N, int iCPU, int nCPU, double *h, double *hu, double *fh, double *fu)
{
  int i, erreur;
  double g=9.81,cm,cmax=0.0,cmaxp=0.0;
  double hg, hd, hug, hud, ug,ud,cg,cd,c1,c2;
  double a_envoyer[2], a_recevoir[2]; //Tableaux pour le transfert de données
  MPI_Status statut;

  for(i=0;i<=N-2;i++)
  {
    hg =  h[i]   ; hd =  h[i+1];
    ug = hu[i]/hg; ud = hu[i+1]/hd;
    calculFlux(i,iCPU,hg,hd,ug,ud,&fh[i],&fu[i],&cmaxp); //Calcul du flux
  }


  if (iCPU%2==0) //Cas pair : on envoie en premier
  {

    if (iCPU != 0) //Tout le monde doit envoyer sauf CPU 0
    {
      a_envoyer[0] = h[0]; a_envoyer[1] = hu[0]; //Données à envoyer au CPU - 1
      //Envoi
      erreur = MPI_Send(a_envoyer , 2 , MPI_DOUBLE_PRECISION, iCPU-1  , 257 + iCPU   , MPI_COMM_WORLD);
    }

    //Réception du CPU + 1
    if (iCPU != nCPU-1) //Tout le monde doit recevoir sauf CPU N-1
    {
      erreur = MPI_Recv(a_recevoir, 2 , MPI_DOUBLE_PRECISION, iCPU+1, 257 + iCPU + 1 ,MPI_COMM_WORLD, &statut);
      hd = a_recevoir[0]; ud = a_recevoir[1]/hd; //Initialisation des paramètres
      hg = h[N-1] ; ug = hu[N-1]/hg;
    }

  }
  else //Cas impair : on reçoit en premier
  {

    if (iCPU != nCPU-1) //Tout le monde doit recevoir sauf CPU N-1
    {
      //Réception
      erreur = MPI_Recv(a_recevoir, 2 , MPI_DOUBLE_PRECISION, iCPU+1, 257 + iCPU + 1 ,MPI_COMM_WORLD, &statut);

      hd = a_recevoir[0]; ud = a_recevoir[1]/hd; //Initialisation des paramètres
      hg = h[N-1] ; ug = hu[N-1]/hg;
    }

    if (iCPU != 0) //Tout le monde doit envoyer sauf CPU 0
    {
      a_envoyer[0] = h[0]; a_envoyer[1] = hu[0]; //Données à envoyer au CPU - 1
      //Envoi
      erreur = MPI_Send(a_envoyer , 2 , MPI_DOUBLE_PRECISION, iCPU-1  , 257 + iCPU   , MPI_COMM_WORLD);
    }

  }

  if (iCPU != nCPU-1) calculFlux(N-1,iCPU,hg,hd,ug,ud,&fh[N-1],&fu[N-1],&cmaxp); //Calcul du flux au point N-1 si on n'est pas CPU N-1

  //On prend le max de toutes les vitesses cmaxp !! Ainsi, tout le monde va avancer du même dt
  erreur = MPI_Allreduce(&cmaxp, &cmax, 1, MPI_DOUBLE_PRECISION, MPI_MAX, MPI_COMM_WORLD);

  return cmax;
}

// ecriture des résultats dans un fichier
// sauvegarde de h et u
void ecrit(int N, FILE* fichier,double *x, double *h, double *hu)
{
  int i;
  for(i=0;i<N;i++)
  {
    fprintf(fichier,"%lf %lf %lf\n",x[i],h[i], hu[i]/h[i]);
  }
  fclose(fichier); // fermeture
}

//fonction principale
int main(int argc, char* argv[])
{
  int iCPU,nCPU,erreur,it;
  int NP, N=1000000, Nt = 1000;
  double x0, dx, cm=0.0, dt, start, end;
  char fichier_init[15], fichier_res[15];
  FILE* finit, *fres;
  double *x,*h, *hu, *fh, *fu;

  erreur = MPI_Init(&argc, &argv);
  erreur = MPI_Comm_size(MPI_COMM_WORLD,&nCPU);
  erreur = MPI_Comm_rank(MPI_COMM_WORLD,&iCPU);

  erreur = MPI_Barrier(MPI_COMM_WORLD);
  start = MPI_Wtime();

  dx = 1.0/((double)N);
  dt = 0.00000001;

  NP = decoupe(iCPU, nCPU, N, dx, &x0); //Découpage du domaine pour chaque CPU
  printf("CPU %d of %d : NP = %d\n",iCPU,nCPU,NP);

  x  = malloc(NP*sizeof(double));  // allocation sur N points
  h  = malloc(NP*sizeof(double));
  fh = malloc(NP*sizeof(double));
  hu = malloc(NP*sizeof(double));
  fu = malloc(NP*sizeof(double));

  init(NP,dx,x,h,hu,x0); //Initialiser le domaine et la condition initiale

  //******* Ecriture des conditions initiales dans fichiers séparés ************
  sprintf(fichier_init,"MPI/init%d.dat",iCPU);
  finit = fopen(fichier_init,"w");
  ecrit(NP,finit,x,h,hu);
  //****************************************************************************

  for(it=1;it<=Nt;it++)   // boucle en temps
  {
    cm = Flux(NP,iCPU,nCPU,h,hu,fh,fu);     // calcul des flux HLL et vitesse d'onde (cm)
    dt = 0.75*dx/cm;              // calcul du pas de temps admissible
    integre(NP,iCPU,nCPU,dt,dx,h,hu,fh,fu); // integration sur un pas de temps Vol. Finis
  }

  //*********** Ecriture du résultat dans fichiers séparés *********************
  sprintf(fichier_res,"MPI/res%d.dat",iCPU);
  fres = fopen(fichier_res,"w");
  ecrit(NP,fres,x,h,hu);
  //****************************************************************************

  erreur = MPI_Barrier(MPI_COMM_WORLD);
  end = MPI_Wtime();

  if (iCPU == 0) printf("Temps maximal : %lf s\n",end-start);

  MPI_Finalize();

  free(x); free(h); free(hu); free(fh); free(fu);

  return 0;
}
