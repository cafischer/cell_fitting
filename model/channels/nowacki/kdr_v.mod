UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
        SUFFIX kdr_v
        USEION k READ ek WRITE ik
        RANGE gbar, ik
		RANGE m_vh, m_vs, m_tau_min, m_tau_max, m_tau_delta
        }

PARAMETER {
        gbar = 0.12 (S/cm2)
		m_vh = 0
        m_vs = 0
        m_tau_min = 0
        m_tau_max = 0
        m_tau_delta = 0
}

STATE {
        m
}

ASSIGNED {
        v (mV)
        ek (mV)
        ik (mA/cm2)
        minf 
	    mtau (ms) 
}

BREAKPOINT {
        SOLVE states METHOD cnexp
	    ik = gbar * m * m * m * m * (v - ek)
}


INITIAL {
	rates(v)
	m = minf
}

DERIVATIVE states {
        rates(v)
        m' =  (minf - m) / mtau
}


PROCEDURE rates(v(mV)) {

UNITSOFF
        minf = 1 / (1 + exp((m_vh - v) / m_vs)) 
	    mtau = m_tau_min + (m_tau_max - m_tau_min) * minf * exp(m_tau_delta * (m_vh - v) / m_vs)
UNITSON
}