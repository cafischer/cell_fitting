TITLE A-type potassium current

COMMENT

animal: Sprague–Dawley rats (Hoffman, 1997)
cell type: pyramidal neuron (Hoffman, 1997)
region: Hippocampus, CA1 (Hoffman, 1997)
current isolation: current subtraction (Hoffman, 1997)
temperature: 24°C
temperature adjustment: activating component Q10 = 5, inactivating component Q10 = 1 (no papers found)
model description: (Migliore, 1999)
experimental data: (Hoffman, 1997)

references:
Hoffman, D. A., Magee, J. C., Colbert, C. M., & Johnston, D. (1997). K&plus; channel regulation of signal propagation in dendrites of hippocampal pyramidal neurons. Nature, 387(6636), 869-875.
Migliore, M., Hoffman, D. A., Magee, J. C., & Johnston, D. (1999). Role of an A-type K+ conductance in the back-propagation of action potentials in the dendrites of hippocampal pyramidal neurons. Journal of computational neuroscience, 7(1), 5-15.

author: Klee Ficker and Heinemann, Migliore

link: http://groups.nbp.northwestern.edu/spruston/sk_models/2stageintegration/2stageintegration_code.zip
(ModelDB #127351)
ENDCOMMENT

NEURON {
        SUFFIX ka
        USEION k READ ek WRITE ik
        RANGE gkabar,gka,ik
        RANGE ninf,linf,taul,taun
        RANGE vhalfn,vhalfl
        GLOBAL lmin,nscale,lscale
}

UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
        (mol) = (1)
}

PARAMETER {
        dt                              (ms)
        v                               (mV)
        ek                              (mV)
        gkabar                          (mho/cm2)

        vhalfn  = 11                    (mV)
        a0n     = 0.05                  (/ms)
        zetan   = -1.5                  (1)
        gmn     = 0.55                  (1)
        pw      = -1                    (1)
        tq      = -40                   (mV)
        qq      = 5                     (mV)
        nmin    = 0.1                   (ms)
        nscale  = 1

        vhalfl  = -56                   (mV)
        a0l     = 0.05                  (/ms)
        zetal   = 3                     (1)
        lmin    = 2                     (ms)
        lscale  = 1

        q10_act = 5
	q10_inact = 1
        temp    = 24                    (degC)
}

STATE {
        n
        l
}

ASSIGNED {
        ik 	(mA/cm2)
        ninf
        linf      
        taul  	(ms)
        taun 	(ms)
        gka 	(mho/cm2)
        tadj_act
	tadj_inact
 	celsius (degC)
}

INITIAL {
        rates(v)
        n=ninf
        l=linf
        gka = gkabar*n*l
        ik = gka*(v-ek)
}        

BREAKPOINT {
        SOLVE states METHOD cnexp
        gka = gkabar*n*l
        ik = gka*(v-ek)
}

DERIVATIVE states {
        rates(v)
        n' = (ninf-n)/taun
        l' = (linf-l)/taul
}

FUNCTION alpn(v(mV)) {
LOCAL zeta
        zeta=zetan+pw/(1+exp((v-tq)/qq))
        alpn = exp(zeta*(v-vhalfn)*1.e-3(V/mV)*9.648e4(coulomb/mol)/(8.315(joule/mol/degC)*(273.16(degC)+celsius))) 
}

FUNCTION betn(v(mV)) {
LOCAL zeta
        zeta=zetan+pw/(1+exp((v-tq)/qq))
        betn = exp(zeta*gmn*(v-vhalfn)*1.e-3(V/mV)*9.648e4(coulomb/mol)/(8.315(joule/mol/degC)*(273.16(degC)+celsius))) 
}


FUNCTION alpl(v(mV)) {
        alpl = exp(zetal*(v-vhalfl)*1.e-3(V/mV)*9.648e4(coulomb/mol)/(8.315(joule/mol/degC)*(273.16(degC)+celsius))) 
}

FUNCTION betl(v(mV)) {
        betl = exp(zetal*(v-vhalfl)*1.e-3(V/mV)*9.648e4(coulomb/mol)/(8.315(joule/mol/degC)*(273.16(degC)+celsius))) 
}


PROCEDURE rates(v (mV)) { 
        LOCAL a, tadj_act, tadj_inact
        tadj_act = q10_act^((celsius-temp)/10 (degC))
        a = alpn(v)
        ninf = 1/(1 + a)
        taun = (betn(v)/(a0n*(1+a))) / tadj_act
        if (taun<nmin) {taun=nmin}
        taun=taun/nscale

	tadj_inact = q10_inact^((celsius-temp)/10 (degC))
        a = alpl(v)
        linf = 1/(1 + a)
        taul = (0.26(ms/mV)*(v+50)) / tadj_inact
        if (taul<lmin) {taul=lmin}
        taul=taul/lscale
}






