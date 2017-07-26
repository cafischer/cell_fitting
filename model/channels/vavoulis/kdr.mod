UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S) = (siemens)
}

NEURON {
	SUFFIX kdr
	USEION k READ ek WRITE ik
	RANGE gbar, ik, n
	RANGE n_vh, n_vs, n_tau_min, n_tau_max, n_tau_delta, n_pow
}

PARAMETER {
    gbar = 0.12 (S/cm2)
	n_pow = 4
	n_vh = 0
    n_vs = 0
    n_tau_min = 0
    n_tau_max = 0
    n_tau_delta = 0
}

STATE {
    n
}

ASSIGNED {
    v (mV)
    ek (mV)
    ik (mA/cm2)
    ninf
	ntau (ms) 
}

BREAKPOINT {
    SOLVE states METHOD cnexp
	ik = gbar * pow(n, n_pow) * (v - ek)
}


INITIAL {
	rates(v)
	n = ninf
}

DERIVATIVE states {
    rates(v)
    n' = (ninf - n) / ntau
}


PROCEDURE rates(v(mV)) {
UNITSOFF
    ninf = 1 / (1 + exp((n_vh - v) / n_vs))
	ntau = n_tau_min + (n_tau_max - n_tau_min) * ninf * exp(n_tau_delta * (n_vh - v) / n_vs)
UNITSON
}
