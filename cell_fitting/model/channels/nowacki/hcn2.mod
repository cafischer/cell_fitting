NEURON {
    SUFFIX hcn2
    NONSPECIFIC_CURRENT i
    RANGE i, gbar, ehcn, n
    RANGE h_vh, h_vs, h_tau_min, h_tau_max, h_tau_delta
}
UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

PARAMETER {
        ehcn = -20
        gbar = 0.12 (S/cm2)
		h_vh = 0
        h_vs = 0
        h_tau_min = 0
        h_tau_max = 0
        h_tau_delta = 0
}

STATE {
        h
}

ASSIGNED {
        v (mV)
        i (mA/cm2)
        hinf
	    htau (ms)
}

BREAKPOINT {
        SOLVE states METHOD cnexp
	    i = gbar * h * (v - ehcn)
}


INITIAL {
	rates(v)
	h = hinf
}

DERIVATIVE states {
        rates(v)
        h' =  (hinf - h) / htau
}

PROCEDURE rates(v(mV)) {

UNITSOFF
    hinf = 1 / (1 + exp((h_vh - v) / h_vs))
	htau = h_tau_min + (h_tau_max - h_tau_min) * hinf * exp(h_tau_delta * (h_vh - v) / h_vs)
UNITSON
}
