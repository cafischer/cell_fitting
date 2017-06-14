UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
        SUFFIX cah
        USEION ca READ eca WRITE ica
        RANGE gbar, ica, m, h
		RANGE m_vh, h_vh, m_vs, h_vs, mtau, htau
        }

PARAMETER {
        gbar = 0.0026 (S/cm2)  : paper: 0.00074
	    m_vh = -15
        m_vs = 5.0
        h_vh = -60
        h_vs = -7.0
		mtau = 0.08
		htau = 300.0
        m0 = 0.0
        h0 = 0.899
}

STATE {
        m h
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
