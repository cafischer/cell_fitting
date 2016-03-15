/* Created by Language version: 6.2.0 */
/* NOT VECTORIZED */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define _threadargscomma_ /**/
#define _threadargs_ /**/
 
#define _threadargsprotocomma_ /**/
#define _threadargsproto_ /**/
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 static double *_p; static Datum *_ppvar;
 
#define t nrn_threads->_t
#define dt nrn_threads->_dt
#define vshift _p[0]
#define gbar _p[1]
#define Oon _p[2]
#define Ooff _p[3]
#define gamma _p[4]
#define delta _p[5]
#define epsilon _p[6]
#define zeta _p[7]
#define x3 _p[8]
#define x4 _p[9]
#define x5 _p[10]
#define x6 _p[11]
#define ina _p[12]
#define g _p[13]
#define C1 _p[14]
#define C2 _p[15]
#define C3 _p[16]
#define C4 _p[17]
#define C5 _p[18]
#define I1 _p[19]
#define I2 _p[20]
#define I3 _p[21]
#define I4 _p[22]
#define I5 _p[23]
#define O _p[24]
#define B _p[25]
#define I6 _p[26]
#define alfac _p[27]
#define btfac _p[28]
#define f01 _p[29]
#define f02 _p[30]
#define f03 _p[31]
#define f04 _p[32]
#define f0O _p[33]
#define fip _p[34]
#define f11 _p[35]
#define f12 _p[36]
#define f13 _p[37]
#define f14 _p[38]
#define f1n _p[39]
#define fi1 _p[40]
#define fi2 _p[41]
#define fi3 _p[42]
#define fi4 _p[43]
#define fi5 _p[44]
#define fin _p[45]
#define b01 _p[46]
#define b02 _p[47]
#define b03 _p[48]
#define b04 _p[49]
#define b0O _p[50]
#define bip _p[51]
#define b11 _p[52]
#define b12 _p[53]
#define b13 _p[54]
#define b14 _p[55]
#define b1n _p[56]
#define bi1 _p[57]
#define bi2 _p[58]
#define bi3 _p[59]
#define bi4 _p[60]
#define bi5 _p[61]
#define bin _p[62]
#define ena _p[63]
#define qt _p[64]
#define DC1 _p[65]
#define DC2 _p[66]
#define DC3 _p[67]
#define DC4 _p[68]
#define DC5 _p[69]
#define DI1 _p[70]
#define DI2 _p[71]
#define DI3 _p[72]
#define DI4 _p[73]
#define DI5 _p[74]
#define DO _p[75]
#define DB _p[76]
#define DI6 _p[77]
#define _g _p[78]
#define _ion_ena	*_ppvar[0]._pval
#define _ion_ina	*_ppvar[1]._pval
#define _ion_dinadv	*_ppvar[2]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_rates(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _p = _prop->param; _ppvar = _prop->dparam;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_narsg", _hoc_setdata,
 "rates_narsg", _hoc_rates,
 0, 0
};
 /* declare global and static user variables */
#define Coff Coff_narsg
 double Coff = 0.5;
#define Con Con_narsg
 double Con = 0.005;
#define alpha alpha_narsg
 double alpha = 150;
#define beta beta_narsg
 double beta = 3;
#define x2 x2_narsg
 double x2 = -20;
#define x1 x1_narsg
 double x1 = 20;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "Con_narsg", "/ms",
 "Coff_narsg", "/ms",
 "alpha_narsg", "/ms",
 "beta_narsg", "/ms",
 "x1_narsg", "mV",
 "x2_narsg", "mV",
 "vshift_narsg", "mV",
 "gbar_narsg", "S/cm2",
 "Oon_narsg", "/ms",
 "Ooff_narsg", "/ms",
 "gamma_narsg", "/ms",
 "delta_narsg", "/ms",
 "epsilon_narsg", "/ms",
 "zeta_narsg", "/ms",
 "x3_narsg", "mV",
 "x4_narsg", "mV",
 "x5_narsg", "mV",
 "x6_narsg", "mV",
 "ina_narsg", "milliamp/cm2",
 "g_narsg", "S/cm2",
 0,0
};
 static double B0 = 0;
 static double C50 = 0;
 static double C40 = 0;
 static double C30 = 0;
 static double C20 = 0;
 static double C10 = 0;
 static double I60 = 0;
 static double I50 = 0;
 static double I40 = 0;
 static double I30 = 0;
 static double I20 = 0;
 static double I10 = 0;
 static double O0 = 0;
 static double delta_t = 0.01;
 static double v = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "Con_narsg", &Con_narsg,
 "Coff_narsg", &Coff_narsg,
 "alpha_narsg", &alpha_narsg,
 "beta_narsg", &beta_narsg,
 "x1_narsg", &x1_narsg,
 "x2_narsg", &x2_narsg,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(_NrnThread*, _Memb_list*, int);
static void _ode_matsol(_NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[3]._i
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "6.2.0",
"narsg",
 "vshift_narsg",
 "gbar_narsg",
 "Oon_narsg",
 "Ooff_narsg",
 "gamma_narsg",
 "delta_narsg",
 "epsilon_narsg",
 "zeta_narsg",
 "x3_narsg",
 "x4_narsg",
 "x5_narsg",
 "x6_narsg",
 0,
 "ina_narsg",
 "g_narsg",
 0,
 "C1_narsg",
 "C2_narsg",
 "C3_narsg",
 "C4_narsg",
 "C5_narsg",
 "I1_narsg",
 "I2_narsg",
 "I3_narsg",
 "I4_narsg",
 "I5_narsg",
 "O_narsg",
 "B_narsg",
 "I6_narsg",
 0,
 0};
 static Symbol* _na_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 79, _prop);
 	/*initialize range parameters*/
 	vshift = 0;
 	gbar = 0.016;
 	Oon = 0.75;
 	Ooff = 0.005;
 	gamma = 150;
 	delta = 40;
 	epsilon = 1.75;
 	zeta = 0.03;
 	x3 = 1e+12;
 	x4 = -1e+12;
 	x5 = 1e+12;
 	x6 = -25;
 	_prop->param = _p;
 	_prop->param_size = 79;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ena */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ina */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dinadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*f)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _narsg_reg() {
	int _vectorized = 0;
  _initlists();
 	ion_reg("na", -10000.);
 	_na_sym = hoc_lookup("na_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 0);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
  hoc_register_prop_size(_mechtype, 79, 4);
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 narsg /media/caro/Daten/Phd/DAP-Project/cell_fitting/model/channels/i686/narsg.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double q10 = 3;
static int _reset;
static char *modelname = "resurgent sodium channel";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(double);
 extern double *_getelm();
 
#define _MATELM1(_row,_col)	*(_getelm(_row + 1, _col + 1))
 
#define _RHS1(_arg) _coef1[_arg + 1]
 static double *_coef1;
 
#define _linmat1  1
 static void* _sparseobj1;
 static void* _cvsparseobj1;
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 
#define _RHS2(arg) _coef2[arg][13]
 static int _slist2[13];static double **_coef2;
 static void seqinitial();
 static int _slist1[13], _dlist1[13]; static double *_temp1;
 static int activation();
 
static int activation ()
 {_reset=0;
 {
   double b_flux, f_flux, _term; int _i;
 {int _i; double _dt1 = 1.0/dt;
for(_i=1;_i<13;_i++){
  	_RHS1(_i) = -_dt1*(_p[_slist1[_i]] - _p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 rates ( _threadargscomma_ v + vshift ) ;
   /* ~ C1 <-> C2 ( f01 , b01 )*/
 f_flux =  f01 * C1 ;
 b_flux =  b01 * C2 ;
 _RHS1( 6) -= (f_flux - b_flux);
 _RHS1( 5) += (f_flux - b_flux);
 
 _term =  f01 ;
 _MATELM1( 6 ,6)  += _term;
 _MATELM1( 5 ,6)  -= _term;
 _term =  b01 ;
 _MATELM1( 6 ,5)  -= _term;
 _MATELM1( 5 ,5)  += _term;
 /*REACTION*/
  /* ~ C2 <-> C3 ( f02 , b02 )*/
 f_flux =  f02 * C2 ;
 b_flux =  b02 * C3 ;
 _RHS1( 5) -= (f_flux - b_flux);
 _RHS1( 4) += (f_flux - b_flux);
 
 _term =  f02 ;
 _MATELM1( 5 ,5)  += _term;
 _MATELM1( 4 ,5)  -= _term;
 _term =  b02 ;
 _MATELM1( 5 ,4)  -= _term;
 _MATELM1( 4 ,4)  += _term;
 /*REACTION*/
  /* ~ C3 <-> C4 ( f03 , b03 )*/
 f_flux =  f03 * C3 ;
 b_flux =  b03 * C4 ;
 _RHS1( 4) -= (f_flux - b_flux);
 _RHS1( 3) += (f_flux - b_flux);
 
 _term =  f03 ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 3 ,4)  -= _term;
 _term =  b03 ;
 _MATELM1( 4 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ C4 <-> C5 ( f04 , b04 )*/
 f_flux =  f04 * C4 ;
 b_flux =  b04 * C5 ;
 _RHS1( 3) -= (f_flux - b_flux);
 _RHS1( 2) += (f_flux - b_flux);
 
 _term =  f04 ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 2 ,3)  -= _term;
 _term =  b04 ;
 _MATELM1( 3 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ C5 <-> O ( f0O , b0O )*/
 f_flux =  f0O * C5 ;
 b_flux =  b0O * O ;
 _RHS1( 2) -= (f_flux - b_flux);
 _RHS1( 12) += (f_flux - b_flux);
 
 _term =  f0O ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 12 ,2)  -= _term;
 _term =  b0O ;
 _MATELM1( 2 ,12)  -= _term;
 _MATELM1( 12 ,12)  += _term;
 /*REACTION*/
  /* ~ O <-> B ( fip , bip )*/
 f_flux =  fip * O ;
 b_flux =  bip * B ;
 _RHS1( 12) -= (f_flux - b_flux);
 _RHS1( 1) += (f_flux - b_flux);
 
 _term =  fip ;
 _MATELM1( 12 ,12)  += _term;
 _MATELM1( 1 ,12)  -= _term;
 _term =  bip ;
 _MATELM1( 12 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ O <-> I6 ( fin , bin )*/
 f_flux =  fin * O ;
 b_flux =  bin * I6 ;
 _RHS1( 12) -= (f_flux - b_flux);
 
 _term =  fin ;
 _MATELM1( 12 ,12)  += _term;
 _term =  bin ;
 _MATELM1( 12 ,0)  -= _term;
 /*REACTION*/
  /* ~ I1 <-> I2 ( f11 , b11 )*/
 f_flux =  f11 * I1 ;
 b_flux =  b11 * I2 ;
 _RHS1( 11) -= (f_flux - b_flux);
 _RHS1( 10) += (f_flux - b_flux);
 
 _term =  f11 ;
 _MATELM1( 11 ,11)  += _term;
 _MATELM1( 10 ,11)  -= _term;
 _term =  b11 ;
 _MATELM1( 11 ,10)  -= _term;
 _MATELM1( 10 ,10)  += _term;
 /*REACTION*/
  /* ~ I2 <-> I3 ( f12 , b12 )*/
 f_flux =  f12 * I2 ;
 b_flux =  b12 * I3 ;
 _RHS1( 10) -= (f_flux - b_flux);
 _RHS1( 9) += (f_flux - b_flux);
 
 _term =  f12 ;
 _MATELM1( 10 ,10)  += _term;
 _MATELM1( 9 ,10)  -= _term;
 _term =  b12 ;
 _MATELM1( 10 ,9)  -= _term;
 _MATELM1( 9 ,9)  += _term;
 /*REACTION*/
  /* ~ I3 <-> I4 ( f13 , b13 )*/
 f_flux =  f13 * I3 ;
 b_flux =  b13 * I4 ;
 _RHS1( 9) -= (f_flux - b_flux);
 _RHS1( 8) += (f_flux - b_flux);
 
 _term =  f13 ;
 _MATELM1( 9 ,9)  += _term;
 _MATELM1( 8 ,9)  -= _term;
 _term =  b13 ;
 _MATELM1( 9 ,8)  -= _term;
 _MATELM1( 8 ,8)  += _term;
 /*REACTION*/
  /* ~ I4 <-> I5 ( f14 , b14 )*/
 f_flux =  f14 * I4 ;
 b_flux =  b14 * I5 ;
 _RHS1( 8) -= (f_flux - b_flux);
 _RHS1( 7) += (f_flux - b_flux);
 
 _term =  f14 ;
 _MATELM1( 8 ,8)  += _term;
 _MATELM1( 7 ,8)  -= _term;
 _term =  b14 ;
 _MATELM1( 8 ,7)  -= _term;
 _MATELM1( 7 ,7)  += _term;
 /*REACTION*/
  /* ~ I5 <-> I6 ( f1n , b1n )*/
 f_flux =  f1n * I5 ;
 b_flux =  b1n * I6 ;
 _RHS1( 7) -= (f_flux - b_flux);
 
 _term =  f1n ;
 _MATELM1( 7 ,7)  += _term;
 _term =  b1n ;
 _MATELM1( 7 ,0)  -= _term;
 /*REACTION*/
  /* ~ C1 <-> I1 ( fi1 , bi1 )*/
 f_flux =  fi1 * C1 ;
 b_flux =  bi1 * I1 ;
 _RHS1( 6) -= (f_flux - b_flux);
 _RHS1( 11) += (f_flux - b_flux);
 
 _term =  fi1 ;
 _MATELM1( 6 ,6)  += _term;
 _MATELM1( 11 ,6)  -= _term;
 _term =  bi1 ;
 _MATELM1( 6 ,11)  -= _term;
 _MATELM1( 11 ,11)  += _term;
 /*REACTION*/
  /* ~ C2 <-> I2 ( fi2 , bi2 )*/
 f_flux =  fi2 * C2 ;
 b_flux =  bi2 * I2 ;
 _RHS1( 5) -= (f_flux - b_flux);
 _RHS1( 10) += (f_flux - b_flux);
 
 _term =  fi2 ;
 _MATELM1( 5 ,5)  += _term;
 _MATELM1( 10 ,5)  -= _term;
 _term =  bi2 ;
 _MATELM1( 5 ,10)  -= _term;
 _MATELM1( 10 ,10)  += _term;
 /*REACTION*/
  /* ~ C3 <-> I3 ( fi3 , bi3 )*/
 f_flux =  fi3 * C3 ;
 b_flux =  bi3 * I3 ;
 _RHS1( 4) -= (f_flux - b_flux);
 _RHS1( 9) += (f_flux - b_flux);
 
 _term =  fi3 ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 9 ,4)  -= _term;
 _term =  bi3 ;
 _MATELM1( 4 ,9)  -= _term;
 _MATELM1( 9 ,9)  += _term;
 /*REACTION*/
  /* ~ C4 <-> I4 ( fi4 , bi4 )*/
 f_flux =  fi4 * C4 ;
 b_flux =  bi4 * I4 ;
 _RHS1( 3) -= (f_flux - b_flux);
 _RHS1( 8) += (f_flux - b_flux);
 
 _term =  fi4 ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 8 ,3)  -= _term;
 _term =  bi4 ;
 _MATELM1( 3 ,8)  -= _term;
 _MATELM1( 8 ,8)  += _term;
 /*REACTION*/
  /* ~ C5 <-> I5 ( fi5 , bi5 )*/
 f_flux =  fi5 * C5 ;
 b_flux =  bi5 * I5 ;
 _RHS1( 2) -= (f_flux - b_flux);
 _RHS1( 7) += (f_flux - b_flux);
 
 _term =  fi5 ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 7 ,2)  -= _term;
 _term =  bi5 ;
 _MATELM1( 2 ,7)  -= _term;
 _MATELM1( 7 ,7)  += _term;
 /*REACTION*/
   /* C1 + C2 + C3 + C4 + C5 + O + B + I1 + I2 + I3 + I4 + I5 + I6 = 1.0 */
 _RHS1(0) =  1.0;
 _MATELM1(0, 0) = 1;
 _RHS1(0) -= I6 ;
 _MATELM1(0, 7) = 1;
 _RHS1(0) -= I5 ;
 _MATELM1(0, 8) = 1;
 _RHS1(0) -= I4 ;
 _MATELM1(0, 9) = 1;
 _RHS1(0) -= I3 ;
 _MATELM1(0, 10) = 1;
 _RHS1(0) -= I2 ;
 _MATELM1(0, 11) = 1;
 _RHS1(0) -= I1 ;
 _MATELM1(0, 1) = 1;
 _RHS1(0) -= B ;
 _MATELM1(0, 12) = 1;
 _RHS1(0) -= O ;
 _MATELM1(0, 2) = 1;
 _RHS1(0) -= C5 ;
 _MATELM1(0, 3) = 1;
 _RHS1(0) -= C4 ;
 _MATELM1(0, 4) = 1;
 _RHS1(0) -= C3 ;
 _MATELM1(0, 5) = 1;
 _RHS1(0) -= C2 ;
 _MATELM1(0, 6) = 1;
 _RHS1(0) -= C1 ;
 /*CONSERVATION*/
   } return _reset;
 }
 
static void seqinitial ()
 {
   zero_matrix(_coef2, 13, 14);
{
  int _counte = -1;
  ++_counte;
 _coef2[_counte][0] -=  1.0 * bi1 ;
 _coef2[_counte][1] -=  1.0 * b01 ;
 _coef2[_counte][2] +=  1.0 * ( fi1 + f01 ) ;
 _RHS2(_counte) -=  0.0 ;
   ;
  ++_counte;
 _coef2[_counte][2] -=  1.0 * f01 ;
 _coef2[_counte][3] -=  1.0 * bi2 ;
 _coef2[_counte][4] -=  1.0 * b02 ;
 _coef2[_counte][1] +=  1.0 * ( b01 + fi2 + f02 ) ;
 _RHS2(_counte) -=  0.0 ;
   ;
  ++_counte;
 _coef2[_counte][1] -=  1.0 * f02 ;
 _coef2[_counte][5] -=  1.0 * bi3 ;
 _coef2[_counte][6] -=  1.0 * b03 ;
 _coef2[_counte][4] +=  1.0 * ( b02 + fi3 + f03 ) ;
 _RHS2(_counte) -=  0.0 ;
   ;
  ++_counte;
 _coef2[_counte][4] -=  1.0 * f03 ;
 _coef2[_counte][7] -=  1.0 * bi4 ;
 _coef2[_counte][8] -=  1.0 * b04 ;
 _coef2[_counte][6] +=  1.0 * ( b03 + fi4 + f04 ) ;
 _RHS2(_counte) -=  0.0 ;
   ;
  ++_counte;
 _coef2[_counte][6] -=  1.0 * f04 ;
 _coef2[_counte][9] -=  1.0 * bi5 ;
 _coef2[_counte][10] -=  1.0 * b0O ;
 _coef2[_counte][8] +=  1.0 * ( b04 + fi5 + f0O ) ;
 _RHS2(_counte) -=  0.0 ;
   ;
  ++_counte;
 _coef2[_counte][8] -=  1.0 * f0O ;
 _coef2[_counte][11] -=  1.0 * bip ;
 _coef2[_counte][12] -=  1.0 * bin ;
 _coef2[_counte][10] +=  1.0 * ( b0O + fip + fin ) ;
 _RHS2(_counte) -=  0.0 ;
   ;
  ++_counte;
 _coef2[_counte][10] -=  1.0 * fip ;
 _coef2[_counte][11] -=  1.0 * bip ;
 _RHS2(_counte) -=  0.0 ;
   ;
  ++_counte;
 _coef2[_counte][2] -=  1.0 * fi1 ;
 _coef2[_counte][3] -=  1.0 * b11 ;
 _coef2[_counte][0] +=  1.0 * ( bi1 + f11 ) ;
 _RHS2(_counte) -=  0.0 ;
   ;
  ++_counte;
 _coef2[_counte][0] -=  1.0 * f11 ;
 _coef2[_counte][1] -=  1.0 * fi2 ;
 _coef2[_counte][5] -=  1.0 * b12 ;
 _coef2[_counte][3] +=  1.0 * ( b11 + bi2 + f12 ) ;
 _RHS2(_counte) -=  0.0 ;
   ;
  ++_counte;
 _coef2[_counte][3] -=  1.0 * f12 ;
 _coef2[_counte][4] -=  1.0 * fi3 ;
 _coef2[_counte][7] -=  1.0 * bi3 ;
 _coef2[_counte][5] +=  1.0 * ( b12 + bi3 + f13 ) ;
 _RHS2(_counte) -=  0.0 ;
   ;
  ++_counte;
 _coef2[_counte][5] -=  1.0 * f13 ;
 _coef2[_counte][6] -=  1.0 * fi4 ;
 _coef2[_counte][9] -=  1.0 * b14 ;
 _coef2[_counte][7] +=  1.0 * ( b13 + bi4 + f14 ) ;
 _RHS2(_counte) -=  0.0 ;
   ;
  ++_counte;
 _coef2[_counte][7] -=  1.0 * f14 ;
 _coef2[_counte][8] -=  1.0 * fi5 ;
 _coef2[_counte][12] -=  1.0 * b1n ;
 _coef2[_counte][9] +=  1.0 * ( b14 + bi5 + f1n ) ;
 _RHS2(_counte) -=  0.0 ;
   ;
  ++_counte;
 _coef2[_counte][2] -=  1.0 ;
 _coef2[_counte][1] -=  1.0 ;
 _coef2[_counte][4] -=  1.0 ;
 _coef2[_counte][6] -=  1.0 ;
 _coef2[_counte][8] -=  1.0 ;
 _coef2[_counte][10] -=  1.0 ;
 _coef2[_counte][11] -=  1.0 ;
 _coef2[_counte][0] -=  1.0 ;
 _coef2[_counte][3] -=  1.0 ;
 _coef2[_counte][5] -=  1.0 ;
 _coef2[_counte][7] -=  1.0 ;
 _coef2[_counte][9] -=  1.0 ;
 _coef2[_counte][12] -=  1.0 ;
 _RHS2(_counte) -=  1.0 ;
   ;
 
}
 }
 
static int  rates (  double _lv ) {
   alfac = pow( ( Oon / Con ) , ( 1.0 / 4.0 ) ) ;
   btfac = pow( ( Ooff / Coff ) , ( 1.0 / 4.0 ) ) ;
   f01 = 4.0 * alpha * exp ( _lv / x1 ) * qt ;
   f02 = 3.0 * alpha * exp ( _lv / x1 ) * qt ;
   f03 = 2.0 * alpha * exp ( _lv / x1 ) * qt ;
   f04 = 1.0 * alpha * exp ( _lv / x1 ) * qt ;
   f0O = gamma * exp ( _lv / x3 ) * qt ;
   fip = epsilon * exp ( _lv / x5 ) * qt ;
   f11 = 4.0 * alpha * alfac * exp ( _lv / x1 ) * qt ;
   f12 = 3.0 * alpha * alfac * exp ( _lv / x1 ) * qt ;
   f13 = 2.0 * alpha * alfac * exp ( _lv / x1 ) * qt ;
   f14 = 1.0 * alpha * alfac * exp ( _lv / x1 ) * qt ;
   f1n = gamma * exp ( _lv / x3 ) * qt ;
   fi1 = Con * qt ;
   fi2 = Con * alfac * qt ;
   fi3 = Con * pow( alfac , 2.0 ) * qt ;
   fi4 = Con * pow( alfac , 3.0 ) * qt ;
   fi5 = Con * pow( alfac , 4.0 ) * qt ;
   fin = Oon * qt ;
   b01 = 1.0 * beta * exp ( _lv / x2 ) * qt ;
   b02 = 2.0 * beta * exp ( _lv / x2 ) * qt ;
   b03 = 3.0 * beta * exp ( _lv / x2 ) * qt ;
   b04 = 4.0 * beta * exp ( _lv / x2 ) * qt ;
   b0O = delta * exp ( _lv / x4 ) * qt ;
   bip = zeta * exp ( _lv / x6 ) * qt ;
   b11 = 1.0 * beta * btfac * exp ( _lv / x2 ) * qt ;
   b12 = 2.0 * beta * btfac * exp ( _lv / x2 ) * qt ;
   b13 = 3.0 * beta * btfac * exp ( _lv / x2 ) * qt ;
   b14 = 4.0 * beta * btfac * exp ( _lv / x2 ) * qt ;
   b1n = delta * exp ( _lv / x4 ) * qt ;
   bi1 = Coff * qt ;
   bi2 = Coff * btfac * qt ;
   bi3 = Coff * pow( btfac , 2.0 ) * qt ;
   bi4 = Coff * pow( btfac , 3.0 ) * qt ;
   bi5 = Coff * pow( btfac , 4.0 ) * qt ;
   bin = Ooff * qt ;
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   _r = 1.;
 rates (  *getarg(1) );
 hoc_retpushx(_r);
}
 
/*CVODE ode begin*/
 static int _ode_spec1() {_reset=0;{
 double b_flux, f_flux, _term; int _i;
 {int _i; for(_i=0;_i<13;_i++) _p[_dlist1[_i]] = 0.0;}
 rates ( _threadargscomma_ v + vshift ) ;
 /* ~ C1 <-> C2 ( f01 , b01 )*/
 f_flux =  f01 * C1 ;
 b_flux =  b01 * C2 ;
 DC1 -= (f_flux - b_flux);
 DC2 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C2 <-> C3 ( f02 , b02 )*/
 f_flux =  f02 * C2 ;
 b_flux =  b02 * C3 ;
 DC2 -= (f_flux - b_flux);
 DC3 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C3 <-> C4 ( f03 , b03 )*/
 f_flux =  f03 * C3 ;
 b_flux =  b03 * C4 ;
 DC3 -= (f_flux - b_flux);
 DC4 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C4 <-> C5 ( f04 , b04 )*/
 f_flux =  f04 * C4 ;
 b_flux =  b04 * C5 ;
 DC4 -= (f_flux - b_flux);
 DC5 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C5 <-> O ( f0O , b0O )*/
 f_flux =  f0O * C5 ;
 b_flux =  b0O * O ;
 DC5 -= (f_flux - b_flux);
 DO += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ O <-> B ( fip , bip )*/
 f_flux =  fip * O ;
 b_flux =  bip * B ;
 DO -= (f_flux - b_flux);
 DB += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ O <-> I6 ( fin , bin )*/
 f_flux =  fin * O ;
 b_flux =  bin * I6 ;
 DO -= (f_flux - b_flux);
 DI6 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ I1 <-> I2 ( f11 , b11 )*/
 f_flux =  f11 * I1 ;
 b_flux =  b11 * I2 ;
 DI1 -= (f_flux - b_flux);
 DI2 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ I2 <-> I3 ( f12 , b12 )*/
 f_flux =  f12 * I2 ;
 b_flux =  b12 * I3 ;
 DI2 -= (f_flux - b_flux);
 DI3 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ I3 <-> I4 ( f13 , b13 )*/
 f_flux =  f13 * I3 ;
 b_flux =  b13 * I4 ;
 DI3 -= (f_flux - b_flux);
 DI4 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ I4 <-> I5 ( f14 , b14 )*/
 f_flux =  f14 * I4 ;
 b_flux =  b14 * I5 ;
 DI4 -= (f_flux - b_flux);
 DI5 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ I5 <-> I6 ( f1n , b1n )*/
 f_flux =  f1n * I5 ;
 b_flux =  b1n * I6 ;
 DI5 -= (f_flux - b_flux);
 DI6 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C1 <-> I1 ( fi1 , bi1 )*/
 f_flux =  fi1 * C1 ;
 b_flux =  bi1 * I1 ;
 DC1 -= (f_flux - b_flux);
 DI1 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C2 <-> I2 ( fi2 , bi2 )*/
 f_flux =  fi2 * C2 ;
 b_flux =  bi2 * I2 ;
 DC2 -= (f_flux - b_flux);
 DI2 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C3 <-> I3 ( fi3 , bi3 )*/
 f_flux =  fi3 * C3 ;
 b_flux =  bi3 * I3 ;
 DC3 -= (f_flux - b_flux);
 DI3 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C4 <-> I4 ( fi4 , bi4 )*/
 f_flux =  fi4 * C4 ;
 b_flux =  bi4 * I4 ;
 DC4 -= (f_flux - b_flux);
 DI4 += (f_flux - b_flux);
 
 /*REACTION*/
  /* ~ C5 <-> I5 ( fi5 , bi5 )*/
 f_flux =  fi5 * C5 ;
 b_flux =  bi5 * I5 ;
 DC5 -= (f_flux - b_flux);
 DI5 += (f_flux - b_flux);
 
 /*REACTION*/
   /* C1 + C2 + C3 + C4 + C5 + O + B + I1 + I2 + I3 + I4 + I5 + I6 = 1.0 */
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE matsol*/
 static int _ode_matsol1() {_reset=0;{
 double b_flux, f_flux, _term; int _i;
   b_flux = f_flux = 0.;
 {int _i; double _dt1 = 1.0/dt;
for(_i=0;_i<13;_i++){
  	_RHS1(_i) = _dt1*(_p[_dlist1[_i]]);
	_MATELM1(_i, _i) = _dt1;
      
} }
 rates ( _threadargscomma_ v + vshift ) ;
 /* ~ C1 <-> C2 ( f01 , b01 )*/
 _term =  f01 ;
 _MATELM1( 6 ,6)  += _term;
 _MATELM1( 5 ,6)  -= _term;
 _term =  b01 ;
 _MATELM1( 6 ,5)  -= _term;
 _MATELM1( 5 ,5)  += _term;
 /*REACTION*/
  /* ~ C2 <-> C3 ( f02 , b02 )*/
 _term =  f02 ;
 _MATELM1( 5 ,5)  += _term;
 _MATELM1( 4 ,5)  -= _term;
 _term =  b02 ;
 _MATELM1( 5 ,4)  -= _term;
 _MATELM1( 4 ,4)  += _term;
 /*REACTION*/
  /* ~ C3 <-> C4 ( f03 , b03 )*/
 _term =  f03 ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 3 ,4)  -= _term;
 _term =  b03 ;
 _MATELM1( 4 ,3)  -= _term;
 _MATELM1( 3 ,3)  += _term;
 /*REACTION*/
  /* ~ C4 <-> C5 ( f04 , b04 )*/
 _term =  f04 ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 2 ,3)  -= _term;
 _term =  b04 ;
 _MATELM1( 3 ,2)  -= _term;
 _MATELM1( 2 ,2)  += _term;
 /*REACTION*/
  /* ~ C5 <-> O ( f0O , b0O )*/
 _term =  f0O ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 12 ,2)  -= _term;
 _term =  b0O ;
 _MATELM1( 2 ,12)  -= _term;
 _MATELM1( 12 ,12)  += _term;
 /*REACTION*/
  /* ~ O <-> B ( fip , bip )*/
 _term =  fip ;
 _MATELM1( 12 ,12)  += _term;
 _MATELM1( 1 ,12)  -= _term;
 _term =  bip ;
 _MATELM1( 12 ,1)  -= _term;
 _MATELM1( 1 ,1)  += _term;
 /*REACTION*/
  /* ~ O <-> I6 ( fin , bin )*/
 _term =  fin ;
 _MATELM1( 12 ,12)  += _term;
 _MATELM1( 0 ,12)  -= _term;
 _term =  bin ;
 _MATELM1( 12 ,0)  -= _term;
 _MATELM1( 0 ,0)  += _term;
 /*REACTION*/
  /* ~ I1 <-> I2 ( f11 , b11 )*/
 _term =  f11 ;
 _MATELM1( 11 ,11)  += _term;
 _MATELM1( 10 ,11)  -= _term;
 _term =  b11 ;
 _MATELM1( 11 ,10)  -= _term;
 _MATELM1( 10 ,10)  += _term;
 /*REACTION*/
  /* ~ I2 <-> I3 ( f12 , b12 )*/
 _term =  f12 ;
 _MATELM1( 10 ,10)  += _term;
 _MATELM1( 9 ,10)  -= _term;
 _term =  b12 ;
 _MATELM1( 10 ,9)  -= _term;
 _MATELM1( 9 ,9)  += _term;
 /*REACTION*/
  /* ~ I3 <-> I4 ( f13 , b13 )*/
 _term =  f13 ;
 _MATELM1( 9 ,9)  += _term;
 _MATELM1( 8 ,9)  -= _term;
 _term =  b13 ;
 _MATELM1( 9 ,8)  -= _term;
 _MATELM1( 8 ,8)  += _term;
 /*REACTION*/
  /* ~ I4 <-> I5 ( f14 , b14 )*/
 _term =  f14 ;
 _MATELM1( 8 ,8)  += _term;
 _MATELM1( 7 ,8)  -= _term;
 _term =  b14 ;
 _MATELM1( 8 ,7)  -= _term;
 _MATELM1( 7 ,7)  += _term;
 /*REACTION*/
  /* ~ I5 <-> I6 ( f1n , b1n )*/
 _term =  f1n ;
 _MATELM1( 7 ,7)  += _term;
 _MATELM1( 0 ,7)  -= _term;
 _term =  b1n ;
 _MATELM1( 7 ,0)  -= _term;
 _MATELM1( 0 ,0)  += _term;
 /*REACTION*/
  /* ~ C1 <-> I1 ( fi1 , bi1 )*/
 _term =  fi1 ;
 _MATELM1( 6 ,6)  += _term;
 _MATELM1( 11 ,6)  -= _term;
 _term =  bi1 ;
 _MATELM1( 6 ,11)  -= _term;
 _MATELM1( 11 ,11)  += _term;
 /*REACTION*/
  /* ~ C2 <-> I2 ( fi2 , bi2 )*/
 _term =  fi2 ;
 _MATELM1( 5 ,5)  += _term;
 _MATELM1( 10 ,5)  -= _term;
 _term =  bi2 ;
 _MATELM1( 5 ,10)  -= _term;
 _MATELM1( 10 ,10)  += _term;
 /*REACTION*/
  /* ~ C3 <-> I3 ( fi3 , bi3 )*/
 _term =  fi3 ;
 _MATELM1( 4 ,4)  += _term;
 _MATELM1( 9 ,4)  -= _term;
 _term =  bi3 ;
 _MATELM1( 4 ,9)  -= _term;
 _MATELM1( 9 ,9)  += _term;
 /*REACTION*/
  /* ~ C4 <-> I4 ( fi4 , bi4 )*/
 _term =  fi4 ;
 _MATELM1( 3 ,3)  += _term;
 _MATELM1( 8 ,3)  -= _term;
 _term =  bi4 ;
 _MATELM1( 3 ,8)  -= _term;
 _MATELM1( 8 ,8)  += _term;
 /*REACTION*/
  /* ~ C5 <-> I5 ( fi5 , bi5 )*/
 _term =  fi5 ;
 _MATELM1( 2 ,2)  += _term;
 _MATELM1( 7 ,2)  -= _term;
 _term =  bi5 ;
 _MATELM1( 2 ,7)  -= _term;
 _MATELM1( 7 ,7)  += _term;
 /*REACTION*/
   /* C1 + C2 + C3 + C4 + C5 + O + B + I1 + I2 + I3 + I4 + I5 + I6 = 1.0 */
 /*CONSERVATION*/
   } return _reset;
 }
 
/*CVODE end*/
 
static int _ode_count(int _type){ return 13;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ena = _ion_ena;
     _ode_spec1 ();
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 13; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  ena = _ion_ena;
 _cvode_sparse(&_cvsparseobj1, 13, _dlist1, _p, _ode_matsol1, &_coef1);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_na_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_na_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_na_sym, _ppvar, 2, 4);
 }

static void initmodel() {
  int _i; double _save;_ninits++;
 _save = t;
 t = 0.0;
{
  B = B0;
  C5 = C50;
  C4 = C40;
  C3 = C30;
  C2 = C20;
  C1 = C10;
  I6 = I60;
  I5 = I50;
  I4 = I40;
  I3 = I30;
  I2 = I20;
  I1 = I10;
  O = O0;
 {
   qt = pow( q10 , ( ( celsius - 22.0 ) / 10.0 ) ) ;
   rates ( _threadargscomma_ v + vshift ) ;
   error =  0; seqinitial();
 error = simeq(13, _coef2, _p, _slist2);
 if(error){fprintf(stderr,"at line 142 in file narsg.mod:\n	SOLVE seqinitial\n"); nrn_complain(_p); abort_run(error);}
 }
  _sav_indep = t; t = _save;

}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  ena = _ion_ena;
 initmodel();
 }}

static double _nrn_current(double _v){double _current=0.;v=_v;{ {
   g = gbar * O ;
   ina = g * ( v - ena ) ;
   }
 _current += ina;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  ena = _ion_ena;
 _g = _nrn_current(_v + .001);
 	{ double _dina;
  _dina = ina;
 _rhs = _nrn_current(_v);
  _ion_dinadv += (_dina - ina)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ina += ina ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type){
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type){
 double _break, _save;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _break = t + .5*dt; _save = t;
 v=_v;
{
  ena = _ion_ena;
 { {
 for (; t < _break; t += dt) {
 error = sparse(&_sparseobj1, 13, _slist1, _dlist1, _p, &t, dt, activation,&_coef1, _linmat1);
 if(error){fprintf(stderr,"at line 134 in file narsg.mod:\n	SOLVE activation METHOD sparse\n"); nrn_complain(_p); abort_run(error);}
 
}}
 t = _save;
 } }}

}

static void terminal(){}

static void _initlists() {
 int _i; static int _first = 1;
  if (!_first) return;
 _slist2[0] = &(I1) - _p;
 _slist2[1] = &(C2) - _p;
 _slist2[2] = &(C1) - _p;
 _slist2[3] = &(I2) - _p;
 _slist2[4] = &(C3) - _p;
 _slist2[5] = &(I3) - _p;
 _slist2[6] = &(C4) - _p;
 _slist2[7] = &(I4) - _p;
 _slist2[8] = &(C5) - _p;
 _slist2[9] = &(I5) - _p;
 _slist2[10] = &(O) - _p;
 _slist2[11] = &(B) - _p;
 _slist2[12] = &(I6) - _p;
 if (_first) _coef2 = makematrix(13, 14);
 _slist1[0] = &(I6) - _p;  _dlist1[0] = &(DI6) - _p;
 _slist1[1] = &(B) - _p;  _dlist1[1] = &(DB) - _p;
 _slist1[2] = &(C5) - _p;  _dlist1[2] = &(DC5) - _p;
 _slist1[3] = &(C4) - _p;  _dlist1[3] = &(DC4) - _p;
 _slist1[4] = &(C3) - _p;  _dlist1[4] = &(DC3) - _p;
 _slist1[5] = &(C2) - _p;  _dlist1[5] = &(DC2) - _p;
 _slist1[6] = &(C1) - _p;  _dlist1[6] = &(DC1) - _p;
 _slist1[7] = &(I5) - _p;  _dlist1[7] = &(DI5) - _p;
 _slist1[8] = &(I4) - _p;  _dlist1[8] = &(DI4) - _p;
 _slist1[9] = &(I3) - _p;  _dlist1[9] = &(DI3) - _p;
 _slist1[10] = &(I2) - _p;  _dlist1[10] = &(DI2) - _p;
 _slist1[11] = &(I1) - _p;  _dlist1[11] = &(DI1) - _p;
 _slist1[12] = &(O) - _p;  _dlist1[12] = &(DO) - _p;
_first = 0;
}
