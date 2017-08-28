UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
        SUFFIX nat_ep
        USEION na READ ena WRITE ina
        RANGE gbar, ina, m, h, n
		RANGE m_vh, m_vs, n_vh, h_vh, n_vs, h_vs, n_tau_min, h_tau_min, n_tau_max, h_tau_max, n_tau_delta, h_tau_delta,
		m_pow, h_pow, a, b
}

PARAMETER {
	gbar = 0.12 (S/cm2)
	a = 1
	b = 0
	m_vh = 0
	m_vs = 0
	m_pow = 3
	h_pow = 1
	n_vh = 0
	h_vh = 0
	n_vs = 0
	h_vs = 0
	n_tau_min = 0
	h_tau_min = 0
	n_tau_max = 0
	h_tau_max = 0
	n_tau_delta = 0
	h_tau_delta = 0
}

STATE {
        n h
}

ASSIGNED {
        v (mV)
        ena (mV)
        ina (mA/cm2)
        ninf
        hinf
	    ntau (ms)
        htau (ms)
		m
		: v_eq
}

BREAKPOINT {
    SOLVE states METHOD cnexp
	ina = gbar * pow(m, m_pow) * pow(h, h_pow) * (v - ena)
}


INITIAL {
	rates(v)
	n = ninf
	h = hinf
}

DERIVATIVE states {
        rates(v)
        n' = (ninf - n) / ntau
        h' = (hinf - h) / htau
}


PROCEDURE rates(v(mV)) {

UNITSOFF
    ninf = 1 / (1 + exp((n_vh - v) / n_vs))
	ntau = n_tau_min + (n_tau_max - n_tau_min) * ninf * exp(n_tau_delta * (n_vh - v) / n_vs)

    hinf = 1 / (1 + exp((h_vh - v) / h_vs))
	htau = h_tau_min + (h_tau_max - h_tau_min) * hinf * exp(h_tau_delta * (h_vh - v) / h_vs)

    : v_eq = - log(1/n - 1) * n_vs + n_vh
    : m = 1 / (1 + exp((-a*v_eq - b + m_vh) / m_vs))
    m = 1 / (1 + exp((a * log(1/n-1) * n_vs - a * n_vh - b + m_vh) / m_vs))
UNITSON
}
