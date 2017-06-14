NEURON {
    SUFFIX hcn
    NONSPECIFIC_CURRENT i
    RANGE i, gbar, ehcn, h
    RANGE h_vh, h_vs, htau
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
        htau = 1
}

STATE {
        h
}

ASSIGNED {
        v (mV)
        i (mA/cm2)
        hinf
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
UNITSON
}
