INDEPENDENT { t FROM 0 TO 1 WITH 1 (ms) }

UNITS { 
	(mV) = (millivolt) 
	(mA) = (milliamp) 
} 
NEURON { 
	SUFFIX caHVA_fit
	USEION ca READ eca WRITE ica
	RANGE gbar, ica
    RANGE a_alpha_h, b_alpha_h, k_alpha_h, a_beta_h, b_beta_h, k_beta_h
    RANGE a_alpha_m, b_alpha_m, k_alpha_m, a_beta_m, b_beta_m, k_beta_m
	}

PARAMETER { 
	gbar = 0.0003 	(mho/cm2)
	v 			    (mV)

	m0 = 0
	h0 = 1

	a_alpha_m = -0.00288
	b_alpha_m = -0.049
	k_alpha_m = 4.63
	a_beta_m = 0.00694
	b_beta_m = 0.447
	k_beta_m = -2.63

	a_alpha_h = -0.00288
	b_alpha_h = -0.049
	k_alpha_h = 4.63
	a_beta_h = 0.00694
	b_beta_h = 0.447
	k_beta_h = -2.63
} 

ASSIGNED { 
	ica 		(mA/cm2)
	eca 		(mV)
	minf
	hinf
    mtau 		(ms)
	htau		(ms)
} 
STATE {
	m
	h
}

BREAKPOINT { 
	SOLVE states METHOD cnexp
	ica = gbar * m * m * h * (v - eca)
} 

INITIAL { 
	settables(v)
	m = m0
	h = h0
} 

DERIVATIVE states { 
	settables(v)
	m' = (minf - m) / mtau
	h' = (hinf - h) / htau
}

UNITSOFF
PROCEDURE settables(v) { 
	LOCAL alpha_m, beta_m, alpha_h, beta_h

    alpha_m = (a_alpha_m * v + b_alpha_m) / (1 - exp((v + b_alpha_m / a_alpha_m) / k_alpha_m))
    beta_m = (a_beta_m * v + b_beta_m) / (1 - exp((v + b_beta_m / a_beta_m) / k_beta_m))

    minf = alpha_m / (alpha_m + beta_m)
    mtau = 1 / (alpha_m + beta_m)

    alpha_h = (a_alpha_h * v + b_alpha_h) / (1 - exp((v + b_alpha_h / a_alpha_h) / k_alpha_h))
    beta_h = (a_beta_h * v + b_beta_h) / (1 - exp((v + b_beta_h / a_beta_h) / k_beta_h))

    hinf = alpha_h / (alpha_h + beta_h)
    htau = 1 / (alpha_h + beta_h)
}
UNITSON
