// my file
//

#include <stdio.h>
#include <stdlib.h>
#include "umfpack.h"
#include "cs.h" // reading triplet matrix
#include <time.h>
#include <omp.h>
#include <assert.h>

#define ABS(x) ((x) >= 0 ? (x) : -(x))

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#ifndef TRUE
#define TRUE (1)
#endif
#ifndef FALSE
#define FALSE (0)
#endif

#define PRINT (0)

/* -------------------------------------------------------------------------- */
/* triplet form of the matrix.  The triplets can be in any order. */
/* -------------------------------------------------------------------------- */

/* static int n = 5, nz = 12 ; */
/* static int Arow [ ] = { 0,  4,  1,  1,   2,   2,  0,  1,  2,  3,  4,  4} ; */
/* static int Acol [ ] = { 0,  4,  0,  2,   1,   2,  1,  4,  3,  2,  1,  2} ; */
/* static double Aval [ ] = {2., 1., 3., 4., -1., -3., 3., 6., 2., 1., 4., 2.} ; */
/* static double b [ ] = {8., 45., -3., 3., 19.}, x [5], r [5] ; */
/* static int Arow [ ] = { 0,  1,  1,  2,   2,   3,  4,  4,  3,  3,  4,  4} ; */
/* static int Acol [ ] = { 0,  1,  0,  0,   2,   3,  0,  4,  1,  4,  3,  2} ; */
/* static double Aval [ ] = {1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.} ; */
/* static double b [ ] = {1., 10., 100., 1., 1.}, x [5], r [5] ; */

typedef struct problem_struct
{
    cs *A ;
    cs *C ;
    int sym ;
    double *x ;
    double *b ;
    double *resid ;
    int *Arow, *Acol;
    double *Aval;
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
    int sym, m, n, mn, nz1, nz2, i ;
    problem *Prob ;
    Prob = cs_calloc (1, sizeof (problem)) ;
    if (!Prob) return (NULL) ;
    
    for (i = 0 ; i < 3 ; i++) fscanf(f, "%*[^\n]\n");

    T = cs_load (f) ;                   /* load triplet matrix T from a file */

    Prob->A = A = cs_compress (T) ;     /* A = compressed-column form of T */
    if (!cs_dupl (A)) return (free_problem (Prob)) ; /* sum up duplicates */
    Prob->sym = sym = is_sym (A) ;      /* determine if A is symmetric */
    m = A->m ; n = A->n ;
    mn = CS_MAX (m,n) ;
    nz1 = A->p [n] ;

    Prob->Arow = (int *) malloc ((nz1) * sizeof (int)) ;
    Prob->Acol = (int *) malloc (nz1 * sizeof (int)) ;
    Prob->Aval = (double *) malloc (nz1 * sizeof (double)) ;
    for (i = 0 ; i < nz1 ; i++) {
      Prob->Arow [ i ]= T->i [ i ];
      Prob->Acol [ i ]= T->p [ i ];
      Prob->Aval [ i ]= T->x [ i ];
      /* printf("%d, %d, %f\n", Prob->Arow [i], Prob->Acol [i], Prob->Aval [i]); */
    }
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

/* -------------------------------------------------------------------------- */
/* error: print a message and exit */
/* -------------------------------------------------------------------------- */

static void error
(
    char *message
)
{
    printf ("\n\n====== error: %s =====\n\n", message) ;
    exit (1) ;
}


/* -------------------------------------------------------------------------- */
/* resid: compute the residual, r = Ax-b or r = A'x=b and return maxnorm (r) */
/* -------------------------------------------------------------------------- */

static double resid
(
    int transpose,
    int n,
    int Ap [ ],
    int Ai [ ],
    double Ax [ ],
    double *x,
    double *b, 
    double *r
)
{
    int i, j, p ;
    double norm ;

    for (i = 0 ; i < n ; i++)
    {
	r [i] = -b [i] ;
    }
    if (transpose)
    {
	for (j = 0 ; j < n ; j++)
	{
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		i = Ai [p] ;
		r [j] += Ax [p] * x [i] ;
	    }
	}
    }
    else
    {
	for (j = 0 ; j < n ; j++)
	{
	    for (p = Ap [j] ; p < Ap [j+1] ; p++)
	    {
		i = Ai [p] ;
		r [i] += Ax [p] * x [j] ;
	    }
	}
    }
    norm = 0. ;
    for (i = 0 ; i < n ; i++)
    {
	norm = MAX (ABS (r [i]), norm) ;
    }
    return (norm) ;
}

/* create a right-hand side */
static void rhs (double *x, double *b, int m)
{
    int i ;
    for (i = 0 ; i < m ; i++) b [i] = 1 + ((double) i) / m ;
    for (i = 0 ; i < m ; i++) x [i] = b [i] ;
}

    
int main (int argc, char **argv)
{

    double Info [UMFPACK_INFO], Control [UMFPACK_CONTROL], *Ax, *Cx, *Lx, *Ux,
	*W, t [2], *Dx, rnorm, *Rb, *y, *Rs, my_t, *Aval ;
    int *Ap, *Ai, *Cp, *Ci, row, col, p, lnz, unz, nr, nc, *Lp, *Li, *Ui, *Up,
	*P, *Q, *Lj, i, j, k, anz, nfr, nchains, *Qinit, fnpiv, lnz1, unz1, nz1,
	status, *Front_npivcol, *Front_parent, *Chain_start, *Wi, *Pinit, n1,
	*Chain_maxrows, *Chain_maxcols, *Front_1strow, *Front_leftmostdesc,
	nzud, do_recip, *Arow, *Acol ;
    void *Symbolic, *Numeric ;

    int n, nz;
    double *b, *x, *r ;

    /* ---------------------------------------------------------------------- */
    /* initializations */
    /* ---------------------------------------------------------------------- */

    umfpack_tic (t) ;

    printf ("\nUMFPACK V%d.%d (%s) demo: _di_ version\n",
	    UMFPACK_MAIN_VERSION, UMFPACK_SUB_VERSION, UMFPACK_DATE) ;

    /* get the default control parameters */
    umfpack_di_defaults (Control) ;

    // Try no re-ordering to preserve L shape of the matrix
    /* Control [UMFPACK_ORDERING] = UMFPACK_ORDERING_NONE ; */
    // no iterative refinement, only single solve
    Control [UMFPACK_IRSTEP] = 0;

    /* change the default print level for this demo */
    /* (otherwise, nothing will print) */
    Control [UMFPACK_PRL] = 6 ;

    /* print the license agreement */
    umfpack_di_report_status (Control, UMFPACK_OK) ;
    Control [UMFPACK_PRL] = 5 ;

    /* print the control parameters */
    umfpack_di_report_control (Control) ;

    /* read matrix */
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
    
    n= (int) Prob->A->n;
    nz= (int) Prob->A->nzmax;
    b= Prob->b;
    x= Prob->x;
    r= Prob->resid;
    Arow = Prob->Arow;
    Acol = Prob->Acol;
    Aval = Prob->Aval;

    printf("# non-zeroes: %d\n", nz);

    printf("%d, %d, %f\n", Arow[0], Acol[0], Aval[0]);
    printf("%d, %d, %f\n", Arow[nz], Acol[nz], Aval[nz]);
    /* status = umfpack_di_report_triplet (n, n, nz, Arow, Acol, Aval, Control) ; */
    /* if (status < 0) */
    /* { */
    /*   umfpack_di_report_status (Control, status) ; */
    /*   error ("umfpack_di_report_triplet failed") ; */
    /* } */

    /* convert to column form */
    nz1 = MAX (nz,1) ;	/* ensure arrays are not of size zero. */
    Ap = (int *) malloc ((n+1) * sizeof (int)) ;
    Ai = (int *) malloc (nz1 * sizeof (int)) ;
    Ax = (double *) malloc (nz1 * sizeof (double)) ;
    if (!Ap || !Ai || !Ax)
    {
	error ("out of memory") ;
    }

    status = umfpack_di_triplet_to_col (n, n, nz, Arow, Acol, Aval,
	Ap, Ai, Ax, (int *) NULL) ;
    if (status < 0)
    {
	umfpack_di_report_status (Control, status) ;
	error ("umfpack_di_triplet_to_col failed") ;
    }

    /* for (i = 0 ; i < n + 1 ; i++) Ap[i] = (int) Prob->A->p[i]; */
    /* for (i = 0 ; i < nz; i++) Ai[i] = (int) Prob->A->i[i]; */
    /* for (i = 0 ; i < nz; i++) Ax[i] = Prob->A->x[i]; */
    if PRINT {
      for (i = 0 ; i < n + 1 ; i++) printf("\n Ap[%d]: %d, ", i, Ap[i]);
      for (i = 0 ; i < nz ; i++) printf("\n Ai[%d]: %d, ", i, Ai[i]);
      for (i = 0 ; i < nz ; i++) printf("\n Ax[%d]: %lf, ", i, Ax[i]);
    }
    
    printf("Ap[0] : %d\n", Ap[0]);
    printf("Ap[n] : %d\n", Ap[n]);
    assert (Ap[0] == 0);
    assert (Ap[n] >= 0);
    for (i = 1 ; i < n + 1 ; i++) {
      assert ( Ap[i] >= Ap[i-1] );
      for (j = Ap [ i-1 ] ; j < Ap [ i ] ; j++) {
        assert (Ai [ j ] < n);
        if ( Ai[ j ] < i-1 ) {
          printf("Ai[j]: %d,i: %d", Ai[j], i-1 ); error( "Not an L marix" ) ;
        }
      }
      for (j = Ap [ i-1 ] + 1 ; j < Ap [ i ] ; j++) {
        if ( Ai [ j ] == Ai [ j-1 ] ) {
          printf("Ai[j]: %d, Ai[j-1]: %d", Ai[j], Ai[j-1] ); error( "Duplicates" ) ;
        }
      }
    }
    /* for (i = 0 ; i < nz ; i++) */
    /* { */ 
    /*   if ( Ai[i] < i ) {printf("Ai[i]: %d,i: %d", Ai[i], i ); error( "Not an L marix" ) ; } */
    /* } // L matrix */
    /* Ap= Prob->A->p; */
    /* Ai= Prob->A->i; */
    /* Ax= Prob->A->x; */

    rhs (x, b, n) ;                             /* compute right-hand side */


    /* ---------------------------------------------------------------------- */
    /* symbolic factorization */
    /* ---------------------------------------------------------------------- */

    printf("Here\n");
    status = umfpack_di_symbolic (n, n, Ap, Ai, Ax, &Symbolic,
	Control, Info) ;
    if (status < 0)
    {
      umfpack_di_report_info (Control, Info) ;
      umfpack_di_report_status (Control, status) ;
      printf ("umfpack_di_symbolic failed, error code: %d", status) ;
      error ("umfpack_di_symbolic failed") ;
    }
    printf("symbolic done\n");

    /* ---------------------------------------------------------------------- */
    /* numeric factorization */
    /* ---------------------------------------------------------------------- */

    status = umfpack_di_numeric (Ap, Ai, Ax, Symbolic, &Numeric,
	Control, Info) ;
    if (status < 0)
    {
	umfpack_di_report_info (Control, Info) ;
	umfpack_di_report_status (Control, status) ;
	error ("umfpack_di_numeric failed") ;
    }

    if PRINT {
      /* print the numeric factorization */
      printf ("\nNumeric factorization of A: ") ;
      (void) umfpack_di_report_numeric (Numeric, Control) ;
    }

    if (umfpack_di_get_lunz (&lnz, &unz, &nr, &nc, &nzud, Numeric) < 0)
    {
      error ("umfpack_di_get_lunz failed") ;
    }

    printf("LU factor statistics: lnz: %d, unz: %d, nr: %d, nc: %d, n: %d, nzud: %d\n", lnz, unz, nr, nc, n, nzud);

    /* ---------------------------------------------------------------------- */
    /* solve Ax=b */
    /* ---------------------------------------------------------------------- */
    double tot_time=0.;
  my_t = tic () ;
    for (i = 0 ; i < n_solve_iter ; i++) {
      status = umfpack_di_solve (UMFPACK_A, Ap, Ai, Ax, x, b,
        Numeric, Control, Info) ;
      tot_time += Info[UMFPACK_SOLVE_TIME];
    }
  my_t= toc(my_t);
 
    umfpack_di_report_info (Control, Info) ;
    /* umfpack_di_report_status (Control, status) ; */
    if (status < 0)
    {
	error ("umfpack_di_solve failed") ;
    }
    printf ("\nx (solution of Ax=b): ") ;
    (void) umfpack_di_report_vector (MIN(10, n), x, Control) ;
    rnorm = resid (FALSE, n, Ap, Ai, Ax, x, b, r) ;
    printf ("maxnorm of residual: %g\n\n", rnorm) ;

  printf("\n UMFPACK_Info, iterative refinement steps take, %f, iterative refinement steps attempted + 1, %f , flops, %8.9f, time, %8.9f, avg reported time (seconds), %8.9f\n", Info[UMFPACK_IR_TAKEN], Info[UMFPACK_IR_ATTEMPTED] + 1, Info[UMFPACK_SOLVE_FLOPS], Info[UMFPACK_SOLVE_TIME], tot_time/n_solve_iter);
  /* printf ("\n n_solve_iter : %d, solve_time (seconds), %8.9f, residual: %g\n", n_solve_iter, my_t, rnorm) ; */
  printf ("\n n_solve_iter : %d, solve_time (seconds), %8.9f, residual, %g, flops, %g\n", n_solve_iter, tot_time, rnorm, Info[UMFPACK_SOLVE_FLOPS]) ;
  free_problem(Prob);
  return 0;
}
