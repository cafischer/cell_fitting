UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
        SUFFIX cat
        USEION ca READ eca WRITE ica
        RANGE gbar, ica, m, h
		RANGE m_vh, h_vh, m_vs, h_vs, mtau, htau
        }

PARAMETER {
        gbar = 0.0006 (S/cm2)
	    m_vh = -54
        m_vs = 5.0
        h_vh = -65
        h_vs = -8.5
		mtau = 2.0
		htau = 32.0
        m0 = 0.014
        h0 = 0.771
}

STATE {
        m
		h
}

ASSIGNED {
        v (mV)
        eca (mV)
        ica (mA/cm2)
        minf
        hinf
}

BREAKPOINT {
        SOLVE states METHOD cnexp
	    ica = gbar * m * m * h * (v - eca)
}


INITIAL {
	rates(v)
	m = m0
	h = h0
}

DERIVATIVE states {
        rates(v)
        m' = (minf - m) / mtau
        h' = (hinf - h) / htau
}


PROCEDURE rates(v(mV)) {

UNITSOFF
        minf = 1 / (1 + exp((m_vh - v) / m_vs))
        hinf = 1 / (1 + exp((h_vh - v) / h_vs))
UNITSON
}
