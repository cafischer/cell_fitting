TITLE delayed-rectifier potassium current

COMMENT

animal: Sprague–Dawley rats (Hoffman, 1997)
cell type: pyramidal neuron (Hoffman, 1997)
region: Hippocampus, CA1 (Hoffman, 1997)
current isolation:  
temperature: 24°C
temperature adjustment: Q10 = 1 (no papers found)
model description: (Migliore, 1999)
experimental data: (Hoffman, 1997)

references:
Hoffman, D. A., Magee, J. C., Colbert, C. M., & Johnston, D. (1997). K&plus; channel regulation of signal propagation in dendrites of hippocampal pyramidal neurons. Nature, 387(6636), 869-875.
Migliore, M., Hoffman, D. A., Magee, J. C., & Johnston, D. (1999). Role of an A-type K+ conductance in the back-propagation of action potentials in the dendrites of hippocampal pyramidal neurons. Journal of computational neuroscience, 7(1), 5-15.

author: Klee Ficker and Heinemann, Migliore

link: http://groups.nbp.northwestern.edu/spruston/sk_models/2stageintegration/2stageintegration_code.zip
(ModelDB #127351)
ENDCOMMENT

UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
        (mol) = (1)

}

NEURON {
        SUFFIX kdr2
        USEION k READ ek WRITE ik
        RANGE gkdr,gbar,ik2
        RANGE ninf,taun
        GLOBAL nscale
}

PARAMETER {
        dt                      (ms)
        v                       (mV)
        ek                      (mV)    : must be explicitely def. in hoc
        celsius                 (degC)
        temp    = 24            (degC)

        gbar = 0.003         (mho/cm2)

        vhalfn  = 13            (mV)
        a0n     = 0.02          (/ms)
        zetan   = -3            (1)
        gmn     = 0.7           (1)

        nmin    = 1             (ms)
        q10     = 1
        nscale  = 1
}

STATE {
        n
}

ASSIGNED {
        ik                      (mA/cm2)
        ik2                     (mA/cm2)
        ninf
        gkdr                    (mho/cm2)
        taun                    (ms)
}

INITIAL {
        rates(v)
        n=ninf
        gkdr = gbar*n
        ik = 0
	ik2 = gkdr*(v-ek)
}        

BREAKPOINT {
        SOLVE states METHOD cnexp
        gkdr = gbar*n
        ik = 0
	ik2 = gkdr*(v-ek)
}

DERIVATIVE states {
        rates(v)
        n' = (ninf-n)/taun
}

FUNCTION alpn(v(mV)) {
        alpn = exp(zetan*(v-vhalfn)*1.e-3(V/mV)*9.648e4(coulomb/mol)/(8.315(joule/degC/mol)*(273.16(degC)+celsius))) 
}

FUNCTION betn(v(mV)) {
        betn = exp(zetan*gmn*(v-vhalfn)*1.e-3(V/mV)*9.648e4(coulomb/mol)/(8.315(joule/degC/mol)*(273.16(degC)+celsius))) 
}

PROCEDURE rates(v (mV)) { :callable from hoc
        LOCAL a,qt
        qt=q10^((celsius-temp)/10(degC))
        a = alpn(v)
        ninf = 1/(1+a)
        taun = betn(v)/(qt*a0n*(1+a))
        if (taun<nmin) {taun=nmin}
        taun=taun/nscale
}




















