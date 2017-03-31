UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
        SUFFIX cav
        USEION ca READ eca WRITE ica
        RANGE gbar, ica
		RANGE m_vh, h_vh, m_vs, h_vs, m_tau_min, h_tau_min, m_tau_max, h_tau_max, m_tau_delta, h_tau_delta, m0, h0
        }

PARAMETER {
        gbar = 0.12 (S/cm2)
		m0 = 0
		h0 = 1
	    m_vh = 0
        h_vh = 0
        m_vs = 0
        h_vs = 0
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
	    mtau (ms)
        htau (ms)
}

BREAKPOINT {
        SOLVE states METHOD cnexp
	    ica = gbar*m*m*m*h*(v - eca)
}


INITIAL {
	rates(v)
	m = m0
	h = h0
}

DERIVATIVE states {
        rates(v)
        m' =  (minf-m)/mtau
        h' = (hinf-h)/htau
}


PROCEDURE rates(v(mV)) {

UNITSOFF
	:"m" sodium activation system
        minf = 1 / (1 + exp((m_vh - v) / m_vs))
	mtau = m_tau_min + (m_tau_max - m_tau_min) * minf * exp(m_tau_delta * (m_vh - v) / m_vs)

	:"h" sodium inactivation system
        hinf = 1 / (1 + exp((h_vh - v) / h_vs))
	htau = h_tau_min + (h_tau_max - h_tau_min) * hinf * exp(h_tau_delta * (h_vh - v) / h_vs)
UNITSON
}
