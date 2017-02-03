COMMENT
17/07/2012
(c) 2012, C. Schmidt-Hieber, University College London
Based on an initial version by Chris Burgess 07/2011

Kinetics based on:
E. Fransen, A. A. Alonso, C. T. Dickson, J. Magistretti, M. E. Hasselmo
Ionic mechanisms in the generation of subthreshold oscillations and
action potential clustering in entorhinal layer II stellate neurons.
Hippocampus 14, 368 (2004).

Do not modify, do not copy and do not redistribute this code.

ENDCOMMENT

NEURON {
    SUFFIX ih_slow
    NONSPECIFIC_CURRENT i
    RANGE i, gbar
    GLOBAL ehcn
    GLOBAL tausn, tausdo, tausdd, tausro, tausrd
    GLOBAL miso, misd, mise
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
    (mA) = (milliamp)
}

PARAMETER {
    gbar = 5.3e-5    (S/cm2)
    ehcn    = -20        (mV)
    tausn   = 5.6        (ms)    : parameters for tau_slow
    tausdo  = 17         (mV)
    tausdd  = 14         (mV)
    tausro    = 260      (mV)
    tausrd    = 43       (mV)
    miso    = 2.83       (mV)    : parameters for steady state m_slow
    misd    = 15.9       (mV)
    mise    = 58.5
}

ASSIGNED {
    v        (mV)
    i        (mA/cm2)
    alphas   (/ms)        : alpha_slow
    betas    (/ms)        : beta_slow
}

INITIAL {
    : assume steady state
    settables(v)
    ms = alphas/(alphas+betas)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    i = gbar*ms *(v-ehcn)
}

STATE {
    ms
}

DERIVATIVE states {
    settables(v)
    ms' = alphas*(1-ms) - betas*ms
}

PROCEDURE settables(v (mV)) {
    LOCAL mis, taus
    TABLE alphas, betas FROM -100 TO 100 WITH 200

    taus = tausn/( exp( (v-tausdo)/tausdd ) + exp( -(v+tausro)/tausrd ) )
    mis = 1/pow( 1 + exp( (v+miso)/misd ), mise )

    alphas = mis/taus
    betas = (1-mis)/taus
}
