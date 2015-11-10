TITLE transient sodium current

COMMENT

references:

: Equations modified by Traub, for Hippocampal Pyramidal cells, in:
: Traub & Miles, Neuronal Networks of the Hippocampus, Cambridge, 1991


: range variable vtraub adjust threshold

ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX Nat
	USEION na READ ena WRITE ina
	USEION k READ ek WRITE ik
	RANGE gnabar, gkbar, vtraub
	RANGE m_inf, h_inf, n_inf
	RANGE tau_m, tau_h, tau_n
	RANGE m_exp, h_exp, n_exp
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
}

PARAMETER {
	gnabar  = .003  (mho/cm2)
	v               (mV)
	ena	        (mV)
	dt              (ms)
	vtraub  = -5   (mV) : (caro) vtraub  = -63   (mV)
	q10_act = 2.7 : TODO
	q10_inact = 1.3 :TODO
	celsius_channel = 21.5 (degC) :TODO 
}

STATE {
	m h n
}

ASSIGNED {
	ina     (mA/cm2)
	ik      (mA/cm2)
	il      (mA/cm2)
	m_inf
	h_inf
	tau_m
	tau_h
	m_exp
	h_exp
	celsius (degC)
	tadj
}

BREAKPOINT {
	SOLVE states
	ina = gnabar * m*m*m*h * (v - ena)
}

:DERIVATIVE states {   : exact Hodgkin-Huxley equations
:       evaluate_fct(v)
:       m' = (m_inf - m) / tau_m
:       h' = (h_inf - h) / tau_h
:}

PROCEDURE states() {    : exact when v held constant
	evaluate_fct(v)
	m = m + m_exp * (m_inf - m)
	h = h + h_exp * (h_inf - h)
	VERBATIM
	return 0;
	ENDVERBATIM
}

UNITSOFF
INITIAL {
	m = 0
	h = 0

:  Q10 was assumed to be 3 for both currents
: original measurements at roomtemperature?

	tadj = 3.0 ^ ((celsius-36)/ 10 )
}

PROCEDURE evaluate_fct(v(mV)) { LOCAL a,b,v2

	v2 = v - vtraub : convert to traub convention

:       a = 0.32 * (13-v2) / ( Exp((13-v2)/4) - 1)
	a = 0.32 * vtrap(13-v2, 4)
:       b = 0.28 * (v2-40) / ( Exp((v2-40)/5) - 1)
	b = 0.28 * vtrap(v2-40, 5)
	tau_m = 1 / (a + b) / tadj
	m_inf = a / (a + b)

	a = 0.128 * Exp((17-v2)/18)
	b = 4 / ( 1 + Exp((40-v2)/5) )
	tau_h = 1 / (a + b) / tadj
	h_inf = a / (a + b)

	m_exp = 1 - Exp(-dt/tau_m)
	h_exp = 1 - Exp(-dt/tau_h)
}
FUNCTION vtrap(x,y) {
	if (fabs(x/y) < 1e-6) {
		vtrap = y*(1 - x/y/2)
	}else{
		vtrap = x/(Exp(x/y)-1)
	}
}

FUNCTION Exp(x) {
	if (x < -100) {
		Exp = 0
	}else{
		Exp = exp(x)
	}
} 

