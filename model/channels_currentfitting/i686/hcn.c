/* Created by Language version: 6.2.0 */
/* VECTORIZED */
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
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
 
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt,
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define gfastbar _p[0]
#define gslowbar _p[1]
#define ehcn _p[2]
#define gslow _p[3]
#define gfast _p[4]
#define i _p[5]
#define i_fast _p[6]
#define i_slow _p[7]
#define mf _p[8]
#define ms _p[9]
#define alphaf _p[10]
#define betaf _p[11]
#define alphas _p[12]
#define betas _p[13]
#define tadj _p[14]
#define Dmf _p[15]
#define Dms _p[16]
#define v _p[17]
#define _g _p[18]
 
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
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_settables(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_ih", _hoc_setdata,
 "settables_ih", _hoc_settables,
 0, 0
};
 
static void _check_settables(double*, Datum*, Datum*, _NrnThread*); 
static void _check_table_thread(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, int _type) {
   _check_settables(_p, _ppvar, _thread, _nt);
 }
 /* declare global and static user variables */
#define celsius_channel celsius_channel_ih
 double celsius_channel = 24;
#define mise mise_ih
 double mise = 58.5;
#define misd misd_ih
 double misd = 15.9;
#define miso miso_ih
 double miso = 2.83;
#define mife mife_ih
 double mife = 1.36;
#define mifd mifd_ih
 double mifd = 9.78;
#define mifo mifo_ih
 double mifo = 74.2;
#define q10 q10_ih
 double q10 = 2.8;
#define tausrd tausrd_ih
 double tausrd = 43;
#define tausro tausro_ih
 double tausro = 260;
#define tausdd tausdd_ih
 double tausdd = 14;
#define tausdo tausdo_ih
 double tausdo = 17;
#define tausn tausn_ih
 double tausn = 5.6;
#define taufrd taufrd_ih
 double taufrd = 52;
#define taufro taufro_ih
 double taufro = 340;
#define taufdd taufdd_ih
 double taufdd = 10;
#define taufdo taufdo_ih
 double taufdo = 1.7;
#define taufn taufn_ih
 double taufn = 0.51;
#define usetable usetable_ih
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_ih", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "celsius_channel_ih", "degC",
 "taufn_ih", "ms",
 "taufdo_ih", "mV",
 "taufdd_ih", "mV",
 "taufro_ih", "mV",
 "taufrd_ih", "mV",
 "tausn_ih", "ms",
 "tausdo_ih", "mV",
 "tausdd_ih", "mV",
 "tausro_ih", "mV",
 "tausrd_ih", "mV",
 "mifo_ih", "mV",
 "mifd_ih", "mV",
 "miso_ih", "mV",
 "misd_ih", "mV",
 "gfastbar_ih", "S/cm2",
 "gslowbar_ih", "S/cm2",
 "ehcn_ih", "mV",
 "gslow_ih", "S/cm2",
 "gfast_ih", "S/cm2",
 "i_ih", "mA/cm2",
 "i_fast_ih", "mA/cm2",
 "i_slow_ih", "mA/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double ms0 = 0;
 static double mf0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "q10_ih", &q10_ih,
 "celsius_channel_ih", &celsius_channel_ih,
 "taufn_ih", &taufn_ih,
 "taufdo_ih", &taufdo_ih,
 "taufdd_ih", &taufdd_ih,
 "taufro_ih", &taufro_ih,
 "taufrd_ih", &taufrd_ih,
 "tausn_ih", &tausn_ih,
 "tausdo_ih", &tausdo_ih,
 "tausdd_ih", &tausdd_ih,
 "tausro_ih", &tausro_ih,
 "tausrd_ih", &tausrd_ih,
 "mifo_ih", &mifo_ih,
 "mifd_ih", &mifd_ih,
 "mife_ih", &mife_ih,
 "miso_ih", &miso_ih,
 "misd_ih", &misd_ih,
 "mise_ih", &mise_ih,
 "usetable_ih", &usetable_ih,
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
 
#define _cvode_ieq _ppvar[0]._i
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "6.2.0",
"ih",
 "gfastbar_ih",
 "gslowbar_ih",
 "ehcn_ih",
 0,
 "gslow_ih",
 "gfast_ih",
 "i_ih",
 "i_fast_ih",
 "i_slow_ih",
 0,
 "mf_ih",
 "ms_ih",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 19, _prop);
 	/*initialize range parameters*/
 	gfastbar = 0;
 	gslowbar = 0;
 	ehcn = -20;
 	_prop->param = _p;
 	_prop->param_size = 19;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 1, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*f)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _hcn_reg() {
	int _vectorized = 1;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_table_reg(_mechtype, _check_table_thread);
  hoc_register_prop_size(_mechtype, 19, 1);
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 ih /media/caro/Daten/Phd/DAP-Project/cell_fitting/model/channels/i686/hcn.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double *_t_alphaf;
 static double *_t_betaf;
 static double *_t_alphas;
 static double *_t_betas;
static int _reset;
static char *modelname = "h-current";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int _f_settables(_threadargsprotocomma_ double);
static int settables(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static void _n_settables(_threadargsprotocomma_ double _lv);
 static int _slist1[2], _dlist1[2];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {int _reset = 0; {
   settables ( _threadargscomma_ v ) ;
   Dmf = alphaf * ( 1.0 - mf ) - betaf * mf ;
   Dms = alphas * ( 1.0 - ms ) - betas * ms ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
 settables ( _threadargscomma_ v ) ;
 Dmf = Dmf  / (1. - dt*( (alphaf)*(( ( - 1.0 ) )) - (betaf)*(1.0) )) ;
 Dms = Dms  / (1. - dt*( (alphas)*(( ( - 1.0 ) )) - (betas)*(1.0) )) ;
 return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) { {
   settables ( _threadargscomma_ v ) ;
    mf = mf + (1. - exp(dt*((alphaf)*(( ( - 1.0 ) )) - (betaf)*(1.0))))*(- ( (alphaf)*(( 1.0 )) ) / ( (alphaf)*(( ( - 1.0) )) - (betaf)*(1.0) ) - mf) ;
    ms = ms + (1. - exp(dt*((alphas)*(( ( - 1.0 ) )) - (betas)*(1.0))))*(- ( (alphas)*(( 1.0 )) ) / ( (alphas)*(( ( - 1.0) )) - (betas)*(1.0) ) - ms) ;
   }
  return 0;
}
 static double _mfac_settables, _tmin_settables;
  static void _check_settables(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  static double _sav_celsius;
  if (!usetable) {return;}
  if (_sav_celsius != celsius) { _maktable = 1;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_settables =  - 100.0 ;
   _tmax =  100.0 ;
   _dx = (_tmax - _tmin_settables)/200.; _mfac_settables = 1./_dx;
   for (_i=0, _x=_tmin_settables; _i < 201; _x += _dx, _i++) {
    _f_settables(_p, _ppvar, _thread, _nt, _x);
    _t_alphaf[_i] = alphaf;
    _t_betaf[_i] = betaf;
    _t_alphas[_i] = alphas;
    _t_betas[_i] = betas;
   }
   _sav_celsius = celsius;
  }
 }

 static int settables(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _lv) { 
#if 0
_check_settables(_p, _ppvar, _thread, _nt);
#endif
 _n_settables(_p, _ppvar, _thread, _nt, _lv);
 return 0;
 }

 static void _n_settables(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _lv){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 _f_settables(_p, _ppvar, _thread, _nt, _lv); return; 
}
 _xi = _mfac_settables * (_lv - _tmin_settables);
 if (isnan(_xi)) {
  alphaf = _xi;
  betaf = _xi;
  alphas = _xi;
  betas = _xi;
  return;
 }
 if (_xi <= 0.) {
 alphaf = _t_alphaf[0];
 betaf = _t_betaf[0];
 alphas = _t_alphas[0];
 betas = _t_betas[0];
 return; }
 if (_xi >= 200.) {
 alphaf = _t_alphaf[200];
 betaf = _t_betaf[200];
 alphas = _t_alphas[200];
 betas = _t_betas[200];
 return; }
 _i = (int) _xi;
 _theta = _xi - (double)_i;
 alphaf = _t_alphaf[_i] + _theta*(_t_alphaf[_i+1] - _t_alphaf[_i]);
 betaf = _t_betaf[_i] + _theta*(_t_betaf[_i+1] - _t_betaf[_i]);
 alphas = _t_alphas[_i] + _theta*(_t_alphas[_i+1] - _t_alphas[_i]);
 betas = _t_betas[_i] + _theta*(_t_betas[_i+1] - _t_betas[_i]);
 }

 
static int  _f_settables ( _threadargsprotocomma_ double _lv ) {
   double _lmif , _lmis , _ltauf , _ltaus ;
 tadj = pow( q10 , ( ( celsius - celsius_channel ) / 10.0 ) ) ;
   _ltauf = 1.0 / tadj * ( taufn / ( exp ( ( _lv - taufdo ) / taufdd ) + exp ( - ( _lv + taufro ) / taufrd ) ) ) ;
   _ltaus = 1.0 / tadj * ( tausn / ( exp ( ( _lv - tausdo ) / tausdd ) + exp ( - ( _lv + tausro ) / tausrd ) ) ) ;
   _lmif = 1.0 / pow ( 1.0 + exp ( ( _lv + mifo ) / mifd ) , mife ) ;
   _lmis = 1.0 / pow ( 1.0 + exp ( ( _lv + miso ) / misd ) , mise ) ;
   alphaf = _lmif / _ltauf ;
   alphas = _lmis / _ltaus ;
   betaf = ( 1.0 - _lmif ) / _ltauf ;
   betas = ( 1.0 - _lmis ) / _ltaus ;
    return 0; }
 
static void _hoc_settables(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; _NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 
#if 1
 _check_settables(_p, _ppvar, _thread, _nt);
#endif
 _r = 1.;
 settables ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 (_p, _ppvar, _thread, _nt);
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol(_NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  int _i; double _save;{
  ms = ms0;
  mf = mf0;
 {
   settables ( _threadargscomma_ v ) ;
   mf = alphaf / ( alphaf + betaf ) ;
   ms = alphas / ( alphas + betas ) ;
   }
 
}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];

#if 0
 _check_settables(_p, _ppvar, _thread, _nt);
#endif
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
 initmodel(_p, _ppvar, _thread, _nt);
}}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gfast = gfastbar * mf ;
   gslow = gslowbar * ms ;
   i = ( gfast + gslow ) * ( v - ehcn ) ;
   i_fast = gfast * ( v - ehcn ) ;
   i_slow = gslow * ( v - ehcn ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type) {
 double _break, _save;
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
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
 { {
 for (; t < _break; t += dt) {
   states(_p, _ppvar, _thread, _nt);
  
}}
 t = _save;
 }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = &(mf) - _p;  _dlist1[0] = &(Dmf) - _p;
 _slist1[1] = &(ms) - _p;  _dlist1[1] = &(Dms) - _p;
   _t_alphaf = makevector(201*sizeof(double));
   _t_betaf = makevector(201*sizeof(double));
   _t_alphas = makevector(201*sizeof(double));
   _t_betas = makevector(201*sizeof(double));
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif
