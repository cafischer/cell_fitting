: Eight state kinetic sodium channel gating scheme
: Modified from k3st.mod, chapter 9.9 (example 9.7)
: of the NEURON book
: 12 August 2008, Christoph Schmidt-Hieber
:
: accompanies the publication:
: Schmidt-Hieber C, Bischofberger J. (2010)
: Fast sodium channel gating supports localized and efficient 
: axonal action potential initiation.
: J Neurosci 30:10233-42

: 11/05/2015 Arnd Roth: add scaling of kinetics with temperature

NEURON {
    SUFFIX na8st2
    USEION na READ ena WRITE ina
    GLOBAL vShift, vShift_inact, maxrate, temp, q10, tadj
    RANGE vShift_inact_local
    RANGE g, gbar, ina2
    RANGE a1_0, a1_1, b1_0, b1_1, a2_0, a2_1
    RANGE b2_0, b2_1, a3_0, a3_1, b3_0, b3_1
    RANGE bh_0, bh_1, bh_2, ah_0, ah_1, ah_2
    RANGE vshift
}

UNITS { (mV) = (millivolt) }

PARAMETER {
    vshift = 0 			 (mV)
    gbar = 33     		 (millimho/cm2)

    a1_0 = 5.142954478051616e+01 (/ms)
    a1_1 = 7.674641248142576e-03 (/mV) 
    
    b1_0 = 9.132202467321037e-03 (/ms)
    b1_1 = 9.342823457307300e-02 (/mV)

    a2_0 = 7.488753944786941e+01 (/ms)
    a2_1 = 2.014613733367395e-02 (/mV) 
    
    b2_0 = 6.387047323688771e-03 (/ms)
    b2_1 = 1.501806374396736e-01 (/mV)

    a3_0 = 3.838866325780059e+01 (/ms)
    a3_1 = 1.253027842782742e-02 (/mV) 
    
    b3_0 = 3.989222258297797e-01 (/ms)
    b3_1 = 9.001475021228642e-02 (/mV)

    bh_0 = 1.687524670388565e+00 (/ms)
    bh_1 = 1.210600094822588e-01 
    bh_2 = 6.827857751079400e-02 (/mV)

    ah_0 = 3.800097357917129e+00 (/ms)
    ah_1 = 4.445911330118979e+03  
    ah_2 = 4.059075804728014e-02 (/mV)

    vShift = 12            (mV)  : shift to the right to account for Donnan potentials
                                 : 12 mV for cclamp, 0 for oo-patch vclamp simulations
    vShift_inact = 10      (mV)  : global additional shift to the right for inactivation
                                 : 10 mV for cclamp, 0 for oo-patch vclamp simulations
    vShift_inact_local = 0 (mV)  : additional shift to the right for inactivation, used as local range variable
    maxrate = 8.00e+03     (/ms) : limiting value for reaction rates
                                 : See Patlak, 1991
    temp = 23 (degC)             : original temperature at which the experiments were done
    q10 = 2.5 (1)                : temperature sensitivity, see Schmidt-Hieber & Bischofberger (2010) Supplemental Figure 8
    celsius (degC)               : nominal temperature of the simulations, set from hoc or python
}

ASSIGNED {
    v    (mV)
    ena  (mV)
    g    (millimho/cm2)
    ina  (milliamp/cm2)
    ina2  (milliamp/cm2)
    a1   (/ms)
    b1   (/ms)
    a2   (/ms)
    b2   (/ms)
    a3   (/ms)
    b3   (/ms)
    ah   (/ms)
    bh   (/ms)
    tadj (1)
}

STATE { c1 c2 c3 i1 i2 i3 i4 o }

BREAKPOINT {
    SOLVE kin METHOD sparse
    g = gbar*o
    ina = 0
    ina2 = g*(v - ena)*(1e-3)
}

INITIAL { SOLVE kin STEADYSTATE sparse }

KINETIC kin {
    : (printf("vshift%f", vshift))
    rates(v+vshift)
    ~ c1 <-> c2 (a1, b1)
    ~ c2 <-> c3 (a2, b2)
    ~ c3 <-> o (a3, b3)
    ~ i1 <-> i2 (a1, b1)
    ~ i2 <-> i3 (a2, b2)
    ~ i3 <-> i4 (a3, b3)
    ~ i1 <-> c1 (ah, bh)
    ~ i2 <-> c2 (ah, bh)
    ~ i3 <-> c3 (ah, bh)
    ~ i4 <-> o  (ah, bh)
    CONSERVE c1 + c2 + c3 + i1 + i2 + i3 + i4 + o = 1
}

: FUNCTION_TABLE tau1(v(mV)) (ms)
: FUNCTION_TABLE tau2(v(mV)) (ms)

PROCEDURE rates(v(millivolt)) {
    LOCAL vS
    vS = v-vShift : note that vShift is subtracted from v, not added to v

    tadj = q10^((celsius - temp)/10)

    a1 = a1_0*exp( a1_1*vS)
    a1 = tadj*a1*maxrate / (a1+maxrate)
    b1 = b1_0*exp(-b1_1*vS)
    b1 = tadj*b1*maxrate / (b1+maxrate)
    
    a2 = a2_0*exp( a2_1*vS)
    a2 = tadj*a2*maxrate / (a2+maxrate)
    b2 = b2_0*exp(-b2_1*vS)
    b2 = tadj*b2*maxrate / (b2+maxrate)
    
    a3 = a3_0*exp( a3_1*vS)
    a3 = tadj*a3*maxrate / (a3+maxrate)
    b3 = b3_0*exp(-b3_1*vS)
    b3 = tadj*b3*maxrate / (b3+maxrate)
    
    bh = bh_0/(1+bh_1*exp(-bh_2*(vS-vShift_inact-vShift_inact_local)))
    bh = tadj*bh*maxrate / (bh+maxrate)
    ah = ah_0/(1+ah_1*exp( ah_2*(vS-vShift_inact-vShift_inact_local)))
    ah = tadj*ah*maxrate / (ah+maxrate)
}
