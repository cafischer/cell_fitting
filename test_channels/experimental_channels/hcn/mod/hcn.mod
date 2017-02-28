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
    SUFFIX hcn
    NONSPECIFIC_CURRENT i
    RANGE i, i_fast, i_slow, gslow, gfast, gslowbar, gfastbar, ehcn
}

UNITS {
    (mV) = (millivolt)
    (S) = (siemens)
    (mA) = (milliamp)
}

PARAMETER {
    q10 = 2.8
    celsius_channel = 24 (degC)
    gfastbar = 0.0    (S/cm2)
    gslowbar = 0.0    (S/cm2)
    ehcn    = -20        (mV) 	 : (Pastoll, 2012)
    vhf = -67.4
    kf = 12.66
    a_alphaf = -0.00289
    b_alphaf = -0.445
    k_alphaf = 24.02
    a_betaf = 0.0271
    b_betaf = -1.024
    k_betaf = -17.4

    vhs = -57.92
    ks = 9.26
    a_alphas = -0.00318
    b_alphas = -0.695
    k_alphas = 26.72
    a_betas = 0.0216
    b_betas = -1.065
    k_betas = -14.25
}

ASSIGNED {
    v        (mV)
    celsius  (degC)
    gslow    (S/cm2)
    gfast    (S/cm2)
    i        (mA/cm2)
    i_fast   (mA/cm2)
    i_slow   (mA/cm2)
    mfinf
    msinf
    mftau
    mstau
    tadj
}

INITIAL {
    : assume steady state
    rates(v)
    mf = mfinf
    ms = msinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    gfast = gfastbar*mf
    gslow = gslowbar*ms
    i = (gfast+gslow)*(v-ehcn)

    i_fast = gfast*(v-ehcn)
    i_slow = gslow*(v-ehcn)
}

STATE {
    mf ms
}

DERIVATIVE states {  
    rates(v)
    mf' = (mfinf-mf)/mftau
    ms' = (msinf-ms)/mstau
}

PROCEDURE rates(v (mV)) { 
    LOCAL alphaf, betaf, alphas, betas

    alphaf = (a_alphaf * v + b_alphaf) / (1 - exp((v + b_alphaf / a_alphaf) / k_alphaf))
    betaf = (a_betaf * v + b_betaf) / (1 - exp((v + b_betaf / a_betaf) / k_betaf))

    mfinf = 1 / (1+exp((v - vhf)/kf))
    mftau = 1 / (alphaf + betaf)

    alphas = (a_alphas * v + b_alphas) / (1 - exp((v + b_alphas / a_alphas) / k_alphas))
    betas = (a_betas * v + b_betas) / (1 - exp((v + b_betas / a_betas) / k_betas))

    msinf = 1 / (1+exp((v - vhs)/ks))
    mstau = 1 / (alphas + betas)
}






