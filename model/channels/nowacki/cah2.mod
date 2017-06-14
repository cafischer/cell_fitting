UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
        SUFFIX cah2
        USEION ca READ eca WRITE ica
        RANGE gbar, ica, m, h
	RANGE m_vh, h_vh, m_vs, h_vs, m_tau_min, h_tau_min, m_tau_max, h_tau_max, m_tau_delta, h_tau_delta
        }

PARAMETER {
        gbar = 0.0026 (S/cm2)  : paper: 0.00074
	    m_vh = -15
        m_vs = 5.0
        h_vh = -60
        h_vs = -7.0
        m_tau_min = 0
        h_tau_min = 0
        m_tau_max = 0
        h_tau_max = 0
        m_tau_delta = 0
        h_tau_delta = 0
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
		mtau
		htau
}

BREAKPOINT {
        SOLVE states METHOD cnexp
	    ica = gbar * m * m * h * (v - eca)
}


INITIAL {
	rates(v)
	m = minf
	h = hinf
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

		mtau = m_tau_min + (m_tau_max - m_tau_min) * minf * exp(m_tau_delta * (m_vh - v) / m_vs)
		htau = h_tau_min + (h_tau_max - h_tau_min) * hinf * exp(h_tau_delta * (h_vh - v) / h_vs)
UNITSON
}
