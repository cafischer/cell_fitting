INDEPENDENT { t FROM 0 TO 1 WITH 1 (ms) }

UNITS { 
	(mV) = (millivolt) 
	(mA) = (milliamp) 
} 
NEURON { 
	SUFFIX nap_fit
	USEION na READ ena WRITE ina
	RANGE gbar, ina
	RANGE htau, hinf, h, minf
    RANGE vh_m, k_m, vh_h, k_h, a_alpha_h, b_alpha_h, k_alpha_h, a_beta_h, b_beta_h, k_beta_h
	}

PARAMETER { 
	gbar = 0.0003 	(mho/cm2)
	v 			    (mV)

	h0 = 1

	vh_m = -44.4
    k_m = -5.2

	a_alpha_h = -0.00288
	b_alpha_h = -0.049
	k_alpha_h = 4.63
	a_beta_h = 0.00694
	b_beta_h = 0.447
	k_beta_h = -2.63
} 

ASSIGNED { 
	ina 		(mA/cm2) 
	ena 		(mV)
	minf 		(1)
	hinf        (1)
	htau		(ms)
} 
STATE {
	h
}

BREAKPOINT { 
	SOLVE states METHOD cnexp
	ina = gbar * minf * h * (v - ena)
} 

INITIAL { 
	settables(v)
	h = h0
} 

DERIVATIVE states { 
	settables(v) 
	h' = (hinf - h) / htau
}
UNITSOFF
 
PROCEDURE settables(v) { 
	LOCAL alpha_h, beta_h

    minf = 1 / (1 + exp((v - vh_m)/k_m))

    alpha_h = (a_alpha_h * v + b_alpha_h) / (1 - exp((v + b_alpha_h / a_alpha_h) / k_alpha_h))
    beta_h = (a_beta_h * v + b_beta_h) / (1 - exp((v + b_beta_h / a_beta_h) / k_beta_h))

    hinf = alpha_h / (alpha_h + beta_h)
    htau = 1 / (alpha_h + beta_h)
}
UNITSON
