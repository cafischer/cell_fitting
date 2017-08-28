NEURON {
    SUFFIX hcn_slow
    NONSPECIFIC_CURRENT i
    RANGE i, gbar, ehcn, n
    RANGE n_vh, n_vs, n_tau_min, n_tau_max, n_tau_delta, n_pow
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(S) = (siemens)
}

PARAMETER {
	ehcn = -20
	gbar = 0.12 (S/cm2)
	n_pow = 1
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
    i (mA/cm2)
    ninf
	ntau (ms) 
}

BREAKPOINT {
    SOLVE states METHOD cnexp
	i = gbar * pow(n, n_pow) * (v - ehcn)
}


INITIAL {
	rates(v)
	n = ninf
}

DERIVATIVE states {
    rates(v)
    n' =  (ninf - n) / ntau
}


PROCEDURE rates(v(mV)) {
UNITSOFF
    ninf = 1 / (1 + exp((n_vh - v) / n_vs))
	ntau = n_tau_min + (n_tau_max - n_tau_min) * ninf * exp(n_tau_delta * (n_vh - v) / n_vs)
UNITSON
}
