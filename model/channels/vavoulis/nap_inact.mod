UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
        SUFFIX nap_inact
        USEION na READ ena WRITE ina
        RANGE gbar, ina, m, h
		RANGE m_vh, h_vh, m_vs, h_vs, h_tau_min, h_tau_max, h_tau_delta
        }

PARAMETER {
        gbar = 0.12 (S/cm2)
	    m_vh = 0
        h_vh = 0
        m_vs = 0
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
        ena (mV)
        ina (mA/cm2)
        minf
        hinf
        htau (ms)
}

BREAKPOINT {
        SOLVE states METHOD cnexp
	    ina = gbar*minf*h*(v - ena)
}


INITIAL {
	rates(v)
	h = hinf
}

DERIVATIVE states {
        rates(v)
        h' = (hinf-h)/htau
}


PROCEDURE rates(v(mV)) {

UNITSOFF
	:"m" sodium activation system
	minf = 1 / (1 + exp((m_vh - v) / m_vs))

	:"h" sodium inactivation system
    hinf = 1 / (1 + exp((h_vh - v) / h_vs))
	htau = h_tau_min + (h_tau_max - h_tau_min) * hinf * exp(h_tau_delta * (h_vh - v) / h_vs)
UNITSON
}
