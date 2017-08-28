UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
        SUFFIX nat
        USEION na READ ena WRITE ina
        RANGE gbar, ina, m, h
		RANGE m_vh, h_vh, m_vs, h_vs, h_a, h_b, h_c, h_d
        }

PARAMETER {
        gbar = 0.065 (S/cm2)
		m_vh = -37
        m_vs = 5.0
        h_vh = -75
        h_vs = -7.0
		h_a = 0.2
		h_b = 0.007
		h_c = -40.6
		h_d = 51.4
		h0 = 0.512
}

STATE {
        m
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
	    ina = gbar * minf * minf * minf * h * (v - ena)
}


INITIAL {
	rates(v)
	h = h0
}

DERIVATIVE states {
        rates(v)
        h' = (hinf - h) / htau
}


PROCEDURE rates(v(mV)) {
UNITSOFF
        minf = 1 / (1 + exp((m_vh - v) / m_vs))
        hinf = 1 / (1 + exp((h_vh - v) / h_vs))
	    htau = h_a + h_b * exp(exp(-(v + h_c)/h_d))
UNITSON
}