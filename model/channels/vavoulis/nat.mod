UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
        SUFFIX nat
        USEION na READ ena WRITE ina
        RANGE gbar, ina
		RANGE m_vh, h_vh, m_vs, h_vs, m_tau_min, h_tau_min, m_tau_max, h_tau_max, m_tau_delta, h_tau_delta
        }

PARAMETER {
        gbar = 0.12 (S/cm2)
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
        ena (mV)
        ina (mA/cm2)
        minf
        hinf
	    mtau (ms)
        htau (ms)
}

BREAKPOINT {
        SOLVE states METHOD cnexp
	    ina = gbar*m*m*m*h*(v - ena)
}


INITIAL {
	rates(v)
	m = minf
	h = hinf
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