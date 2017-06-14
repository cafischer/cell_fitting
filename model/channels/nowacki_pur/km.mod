UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
        SUFFIX km
        USEION k READ ek WRITE ik
        RANGE gbar, ik, m
		RANGE m_vh, m_vs, mtau
        }

PARAMETER {
        gbar = 0.0008 (S/cm2)
		m_vh = -30
        m_vs = 10.0
		mtau = 75.0
		m0 = 0.011
}

STATE {
        m
}

ASSIGNED {
        v (mV)
        ek (mV)
        ik (mA/cm2)
        minf
}

BREAKPOINT {
        SOLVE states METHOD cnexp
	    ik = gbar * m * (v - ek)
}


INITIAL {
	rates(v)
	m = m0
}

DERIVATIVE states {
        rates(v)
		m' = (minf - m) / mtau
}


PROCEDURE rates(v(mV)) {
UNITSOFF
		minf = 1 / (1 + exp((m_vh - v) / m_vs))
UNITSON
}