TITLE h-current

COMMENT

animal: Long-Evans rats (Dickson, 2000)
cell type: stellate cell (Dickson, 2000)
region: Entorhinal Cortex layer 2 (Dickson, 2000)
current isolation: ZD7288 current subtraction (Dickson, 2000)
temperature: 24°C (Dickson, 2000)
temperature adjustment: Q10 = 2.8 (Nolan, 2007)
model description: (Fransen et al., 2004)
experimental data: (Dickson, 2000)

references:
Fransén, E., Alonso, A. A., Dickson, C. T., Magistretti, J., & Hasselmo, M. E. (2004). Ionic mechanisms in the generation of subthreshold oscillations and action potential clustering in entorhinal layer II stellate neurons. Hippocampus, 14(3), 368-384.
Dickson, C. T., Magistretti, J., Shalinsky, M. H., Fransén, E., Hasselmo, M. E., & Alonso, A. (2000). Properties and role of I h in the pacing of subthreshold oscillations in entorhinal cortex layer II neurons. Journal of Neurophysiology, 83(5), 2562-2579.
Nolan, M. F., Dudman, J. T., Dodson, P. D., & Santoro, B. (2007). HCN1 channels control resting and active integrative properties of stellate cells from layer II of the entorhinal cortex. The Journal of Neuroscience, 27(46), 12440-12451.

link: http://senselab.med.yale.edu/modeldb/showmodel.cshtml?model=150239&file=\grid\nrn\mod\hcn.mod

author: Chris Burgess, Schmidt-Hieber

ENDCOMMENT

NEURON {
    SUFFIX ih
    NONSPECIFIC_CURRENT i
    RANGE i, gslow, gfast, gslowbar, gfastbar, ehcn
    GLOBAL taufn, taufdo, taufdd, taufro, taufrd
    GLOBAL tausn, tausdo, tausdd, tausro, tausrd
    GLOBAL mifo, mifd, mife, miso, misd, mise
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
    (mA) = (milliamp)
}

PARAMETER {
    q10 = 2.8
    celsius_channel = 24 (degC)
    gfastbar = 9.8e-5    (S/cm2)
    gslowbar = 5.3e-5    (S/cm2)
    ehcn    = -20        (mV) 	 : (Pastoll, 2012)
    taufn   = 0.51       (ms)    : parameters for tau_fast
    taufdo  = 1.7        (mV)
    taufdd  = 10         (mV)
    taufro  = 340        (mV)
    taufrd  = 52         (mV)
    tausn   = 5.6        (ms)    : parameters for tau_slow
    tausdo  = 17         (mV)
    tausdd  = 14         (mV)
    tausro  = 260        (mV)
    tausrd  = 43         (mV)
    mifo    = 74.2       (mV)    : parameters for steady state m_fast
    mifd    = 9.78       (mV)
    mife    = 1.36
    miso    = 2.83       (mV)    : parameters for steady state m_slow
    misd    = 15.9       (mV)
    mise    = 58.5
}

ASSIGNED {
    v        (mV)
    celsius  (degC)
    gslow    (S/cm2)
    gfast    (S/cm2)
    i        (mA/cm2)
    alphaf   (/ms)        : alpha_fast
    betaf    (/ms)        : beta_fast
    alphas   (/ms)        : alpha_slow
    betas    (/ms)        : beta_slow
    tadj
}

INITIAL {
    : assume steady state
    settables(v)
    mf = alphaf/(alphaf+betaf)
    ms = alphas/(alphas+betas)
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    gfast = gfastbar*mf
    gslow = gslowbar*ms
    i = (gfast+gslow)*(v-ehcn)
}

STATE {
    mf ms
}

DERIVATIVE states {  
    settables(v)
    mf' = alphaf*(1-mf) - betaf*mf
    ms' = alphas*(1-ms) - betas*ms
}

PROCEDURE settables(v (mV)) { 
    LOCAL mif, mis, tauf, taus
    TABLE alphaf, betaf, alphas, betas DEPEND celsius FROM -100 TO 100 WITH 200

    tadj = q10 ^ ((celsius - celsius_channel) / 10 (degC))

    tauf = 1/tadj * (taufn/(exp((v-taufdo)/taufdd) + exp(-(v+taufro)/taufrd)))
    taus = 1/tadj * (tausn/(exp((v-tausdo)/tausdd) + exp(-(v+tausro)/tausrd)))
    mif = 1/pow( 1 + exp( (v+mifo)/mifd ), mife )
    mis = 1/pow( 1 + exp( (v+miso)/misd ), mise )

    alphaf = mif/tauf
    alphas = mis/taus
    betaf = (1-mif)/tauf
    betas = (1-mis)/taus
}






