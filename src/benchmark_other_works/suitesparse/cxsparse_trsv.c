#include "cs.h"
#include <time.h>
#include<stdio.h>
#include<stdlib.h>
#include <assert.h>
#include <omp.h>

typedef struct problem_struct
{
    cs *A ;
    cs *C ;
    int sym ;
    double *x ;
    double *b ;
    double *resid ;
} problem ;

problem *get_problem (FILE *f, double tol) ;
problem *free_problem (problem *Prob) ;

/* 1 if A is square & upper tri., -1 if square & lower tri., 0 otherwise */
static int is_sym (cs *A)
{
    int is_upper, is_lower, j, p, n = A->n, m = A->m, *Ap = A->p, *Ai = A->i ;
    if (m != n) return (0) ;
    is_upper = 1 ;
    is_lower = 1 ;
    for (j = 0 ; j < n ; j++)
    {
        for (p = Ap [j] ; p < Ap [j+1] ; p++)
        {
            if (Ai [p] > j) is_upper = 0 ;
            if (Ai [p] < j) is_lower = 0 ;
        }
    }
    return (is_upper ? 1 : (is_lower ? -1 : 0)) ;
}


/* true for off-diagonal entries */
static int dropdiag (int i, int j, double aij, void *other) { return (i != j) ;}

/* C = A + triu(A,1)' */
static cs *make_sym (cs *A)
{
    cs *AT, *C ;
    AT = cs_transpose (A, 1) ;          /* AT = A' */
    cs_fkeep (AT, &dropdiag, NULL) ;    /* drop diagonal entries from AT */
    C = cs_add (A, AT, 1, 1) ;          /* C = A+AT */
    cs_spfree (AT) ;
    return (C) ;
}

problem *get_problem (FILE *f, double tol)
{
    cs *T, *A, *C ;
    int sym, m, n, mn, nz1, nz2 ;
    problem *Prob ;
    Prob = cs_calloc (1, sizeof (problem)) ;
    if (!Prob) return (NULL) ;
    
    int i;
    for (i = 0 ; i < 3 ; i++) fscanf(f, "%*[^\n]\n");

    T = cs_load (f) ;                   /* load triplet matrix T from a file */
    Prob->A = A = cs_compress (T) ;     /* A = compressed-column form of T */
    cs_spfree (T) ;                     /* clear T */
    if (!cs_dupl (A)) return (free_problem (Prob)) ; /* sum up duplicates */
    Prob->sym = sym = is_sym (A) ;      /* determine if A is symmetric */
    m = A->m ; n = A->n ;
    mn = CS_MAX (m,n) ;
    nz1 = A->p [n] ;
    cs_dropzeros (A) ;                  /* drop zero entries */
    nz2 = A->p [n] ;
    if (tol > 0) cs_droptol (A, tol) ;  /* drop tiny entries (just to test) */
    Prob->C = C = sym ? make_sym (A) : A ;  /* C = A + triu(A,1)', or C=A */
    if (!C) return (free_problem (Prob)) ;
    printf ("\n--- Matrix: %g-by-%g, nnz: %g (sym: %g: nnz %g), norm: %8.2e\n",
            (double) m, (double) n, (double) (A->p [n]), (double) sym,
            (double) (sym ? C->p [n] : 0), cs_norm (C)) ;
    if (nz1 != nz2) printf ("zero entries dropped: %g\n", (double) (nz1 - nz2));
    if (nz2 != A->p [n]) printf ("tiny entries dropped: %g\n",
            (double) (nz2 - A->p [n])) ;
    Prob->b = cs_malloc (mn, sizeof (double)) ;
    Prob->x = cs_malloc (mn, sizeof (double)) ;
    Prob->resid = cs_malloc (mn, sizeof (double)) ;
    return ((!Prob->b || !Prob->x || !Prob->resid) ? free_problem (Prob) : Prob) ;
}

/* free a problem */
problem *free_problem (problem *Prob)
{
    if (!Prob) return (NULL) ;
    cs_spfree (Prob->A) ;
    if (Prob->sym) cs_spfree (Prob->C) ;
    cs_free (Prob->b) ;
    cs_free (Prob->x) ;
    cs_free (Prob->resid) ;
    return (cs_free (Prob)) ;
}

;

static double tic (void) { return omp_get_wtime() ; }
static double toc (double t) { return omp_get_wtime() - t ; }

/* static double tic (void) { return (clock () / (double) CLOCKS_PER_SEC) ; } */
/* static double toc (double t) { double s = tic () ; return (CS_MAX (0, s-t)) ; } */

/* create a right-hand side */
static void rhs (double *x, double *b, int m)
{
    int i ;
    for (i = 0 ; i < m ; i++) b [i] = 1 + ((double) i) / m ;
    for (i = 0 ; i < m ; i++) x [i] = b [i] ;
}

int main(int argc, char* argv[]) {
  cs *A, *C= NULL;
  double *b, *x, *resid, *y, t;
  int n, k;
  css *S = NULL ;
  csn *N = NULL ;

  // args
  int nrequired_args = 4;
  if (argc != nrequired_args){
      fprintf(stderr, "improper arguments");
      exit(1);
  }
  /** parse arguments */
  int iarg = 1;
  char* strpathA = argv[iarg];    iarg++;
  int nthreads = atoi(argv[iarg]);    iarg++;
  int n_solve_iter = atoi(argv[iarg]);    iarg++;
  assert(nrequired_args == iarg);
  
  FILE *fp= fopen(strpathA, "r");
  problem *Prob= get_problem(fp, 0);

  /* assert ( Prob->sym != 1 ); */

  A = Prob->A ; C = Prob->C ; b = Prob->b ; x = Prob->x ; resid = Prob->resid;
  n = A->n ;
  rhs (x, b, n) ;                             /* compute right-hand side */

  y = cs_malloc (n, sizeof (double)) ;
  int i,j= 0;
  for (j = 0 ; j < n ; j++) y[j] = b[j] ;
  /* for (i = 0 ; i < 50 ; i++) printf("%f ", y[i]) ; */
  t = tic () ;
  for (i = 0 ; i < n_solve_iter ; i++) cs_lsolve (A, y) ;                       /* y = L\y */
  printf ("\n solve_time (seconds), %8.9f,\n", toc (t)) ;
  for (i = 0 ; i < 50 ; i++) printf("%f ", y[i]) ;
  printf("\n");
  return(0);
}
