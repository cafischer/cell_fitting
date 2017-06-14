UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
        SUFFIX ka
        USEION k READ ek WRITE ik
        RANGE gbar, ik, m, h
		RANGE m_vh, m_vs, m_tau_min, m_tau_max, m_tau_delta, h_vh, h_vs, h_tau_min, h_tau_max, h_tau_delta
        }

PARAMETER {
        gbar = 0.12 (S/cm2)
		m_vh = 0
        m_vs = 0
        m_tau_min = 0
        m_tau_max = 0
        m_tau_delta = 0
		h_vh = 0
        h_vs = 0
        h_tau_min = 0
        h_tau_max = 0
        h_tau_delta = 0
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
	    mtau (ms)
        hinf
	    htau (ms)
}

BREAKPOINT {
        SOLVE states METHOD cnexp
	    ik = gbar * m *  h * (v - ek)
}


INITIAL {
	rates(v)
	m = minf
    h = hinf
}

DERIVATIVE states {
        rates(v)
        m' =  (minf - m) / mtau
        h' =  (hinf - h) / htau
}


PROCEDURE rates(v(mV)) {
UNITSOFF
        minf = 1 / (1 + exp((m_vh - v) / m_vs))
	    mtau = m_tau_min + (m_tau_max - m_tau_min) * minf * exp(m_tau_delta * (m_vh - v) / m_vs)

        hinf = 1 / (1 + exp((h_vh - v) / h_vs))
	    htau = h_tau_min + (h_tau_max - h_tau_min) * hinf * exp(h_tau_delta * (h_vh - v) / h_vs)
UNITSON
}