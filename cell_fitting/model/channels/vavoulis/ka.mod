UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
    SUFFIX ka
	USEION k READ ek WRITE ik
    RANGE gbar, ik, n, l
	RANGE n_vh, n_vs, n_tau_min, n_tau_max, n_tau_delta, l_vh, l_vs, l_tau_min, l_tau_max, l_tau_delta
        }

PARAMETER {
    gbar = 0.12 (S/cm2)
	n_vh = 0
    n_vs = 0
    n_tau_min = 0
    n_tau_max = 0
    n_tau_delta = 0
	l_vh = 0
    l_vs = 0
    l_tau_min = 0
    l_tau_max = 0
    l_tau_delta = 0
}

STATE {
        n
        l
}

ASSIGNED {
        v (mV)
        ek (mV)
        ik (mA/cm2)
        ninf
	    ntau (ms)
        linf
	    ltau (ms)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik = gbar*n*l*(v - ek)
}


INITIAL {
    rates(v)
    n = ninf
    l = linf
}

DERIVATIVE states {
        rates(v)
        n' =  (ninf-n)/ntau
        l' =  (linf-l)/ltau
}


PROCEDURE rates(v(mV)) {

UNITSOFF
        ninf = 1 / (1 + exp((n_vh - v) / n_vs))
	    ntau = n_tau_min + (n_tau_max - n_tau_min) * ninf * exp(n_tau_delta * (n_vh - v) / n_vs)

        linf = 1 / (1 + exp((l_vh - v) / l_vs))
	    ltau = l_tau_min + (l_tau_max - l_tau_min) * linf * exp(l_tau_delta * (l_vh - v) / l_vs)

UNITSON
}
