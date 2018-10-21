
NEURON {
    SUFFIX hcn_fit
    NONSPECIFIC_CURRENT i
    RANGE i, i_fast, i_slow, gslow, gfast, gslowbar, gfastbar, ehcn
    RANGE a_alpha_h, b_alpha_h, k_alpha_h, a_beta_h, b_beta_h, k_beta_h
    RANGE a_alpha_m, b_alpha_m, k_alpha_m, a_beta_m, b_beta_m, k_beta_m
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
    (mA) = (milliamp)
}

PARAMETER {
    gfastbar = 0.0    (S/cm2)
    gslowbar = 0.0    (S/cm2)
    ehcn    = -20        (mV)

	m0 = 0
	h0 = 0

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
    v        (mV)
    gslow    (S/cm2)
    gfast    (S/cm2)
    i        (mA/cm2)
    i_fast   (mA/cm2)
    i_slow   (mA/cm2)
    minf
    hinf
    mtau
    htau
}

INITIAL {
    rates(v)
    m = m0
    h = h0
    }

BREAKPOINT {
    SOLVE states METHOD cnexp
    gfast = gfastbar*m
    gslow = gslowbar*h
    i = (gfast+gslow)*(v-ehcn)

    i_fast = gfast*(v-ehcn)
    i_slow = gslow*(v-ehcn)
}

STATE {
    m
    h
}

DERIVATIVE states {  
    rates(v)
    m' = (minf-m) / mtau
    h' = (hinf-h) / htau
}

PROCEDURE rates(v (mV)) { 
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






