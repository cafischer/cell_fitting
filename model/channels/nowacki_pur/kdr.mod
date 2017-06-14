UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
        SUFFIX kdr
        USEION k READ ek WRITE ik
        RANGE gbar, ik, m, h
		RANGE m_vh, m_vs, h_vh, h_vs, mtau, htau
        }

PARAMETER {
        gbar = 0.0095 (S/cm2)
		m_vh = -5.8
        m_vs = 11.4
		h_vh = -68
        h_vs = -9.7
		mtau = 1.0
		htau = 1400
		m0 = 0.002
        h0 = 0.68
}

STATE {
        m
  		h
}

ASSIGNED {
        v (mV)
        ek (mV)
        ik (mA/cm2)
        minf
		hinf
}

BREAKPOINT {
        SOLVE states METHOD cnexp
	    ik = gbar * m * h * (v - ek)
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