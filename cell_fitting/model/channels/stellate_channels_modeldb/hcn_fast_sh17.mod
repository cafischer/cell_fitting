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
    SUFFIX hcn_fast_sh17
    NONSPECIFIC_CURRENT i
    RANGE i, gfast, gbar
    GLOBAL ehcn
    GLOBAL taufn, taufdo, taufdd, taufro, taufrd
    GLOBAL mifo, mifd, mife
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
    (mA) = (milliamp)
}

PARAMETER {
    gbar = 9.8e-5    (S/cm2)
    ehcn    = -20        (mV)
    taufn   = 0.51       (ms)    : original: .51 parameters for tau_fast
    taufdo  = 1.7        (mV)
    taufdd  = 10         (mV)
    taufro    = 340      (mV)
    taufrd    = 52       (mV)
    mifo    = 74.2       (mV)    : parameters for steady state m_fast
    mifd    = 9.78       (mV)
    mife    = 1.36
}

ASSIGNED {
    v        (mV)
    gfast    (S/cm2)
    i        (mA/cm2)
    alphaf   (/ms)        : alpha_fast
    betaf    (/ms)        : beta_fast
}

INITIAL {
    : assume steady state
    settables(v)
    mf = alphaf/(alphaf+betaf)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    gfast = gbar*mf
    i = gfast * (v-ehcn)
}

STATE {
    mf
}

DERIVATIVE states {  
    settables(v)
    mf' = alphaf*(1-mf) - betaf*mf
}

PROCEDURE settables(v (mV)) { 
    LOCAL mif, tauf 
    TABLE alphaf, betaf FROM -100 TO 100 WITH 200

    tauf = taufn/( exp( (v-taufdo)/taufdd ) + exp( -(v+taufro)/taufrd ) )
    mif = 1/pow( 1 + exp( (v+mifo)/mifd ), mife )

    alphaf = mif/tauf
    betaf = (1-mif)/tauf
}
