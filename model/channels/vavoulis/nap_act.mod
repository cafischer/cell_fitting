UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
        SUFFIX nap_act
        USEION na READ ena WRITE ina
        RANGE gbar, ina, minf
		RANGE m_vh, m_vs
        }

PARAMETER {
        gbar = 0.12 (S/cm2)
	m_vh = 10
        m_vs = 15
}

ASSIGNED {
        v (mV)
        ena (mV)
        ina (mA/cm2)
        minf
}

BREAKPOINT {
		minf = 1 / (1 + exp((m_vh - v) / m_vs))
	    ina = gbar * minf * (v - ena)
}
