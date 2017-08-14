NEURON {
    SUFFIX nap_markov
    USEION na READ ena WRITE ina
    RANGE gbar, ina
    RANGE a1_0, a1_1, a2_0, a2_1, a3_0, a3_1, b1_0, b1_1, b2_0, b2_1, b3_0, b3_1
}

UNITS { 
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gbar = 0.1  (S/cm2)

    a1_0 = 10 (/ms)
    a1_1 = 0.001 (/mV)
    a2_0 = 10 (/ms)
    a2_1 = 0.001 (/mV)
    a3_0 = 10 (/ms)
    a3_1 = 0.001 (/mV)
    
    b1_0 = 0.001 (/ms)
    b1_1 = 0.1 (/mV)
    b2_0 = 0.001 (/ms)
    b2_1 = 0.1 (/mV)
    b3_0 = 0.001 (/ms)
    b3_1 = 0.1 (/mV)
}

ASSIGNED {
    v    (mV)
    ena  (mV)
    ina  (mA/cm2)
    a1   (/ms)
    a2   (/ms)
    a3   (/ms)
    b1   (/ms)
    b2   (/ms)
    b3   (/ms)
}

STATE {
    c i o
}

BREAKPOINT {
    SOLVE kin METHOD sparse
    ina = gbar * o * (v - ena)
}

INITIAL {
    SOLVE kin STEADYSTATE sparse
}

KINETIC kin {
    rates(v)
    ~ c <-> o (a1, b1)
    ~ i <-> c (a2, b2)
    ~ i <-> o (a3, b3)
    CONSERVE c + i + o = 1
}

PROCEDURE rates(vm (millivolt)) {
    a1 = a1_0 * exp(a1_1 * vm)
    a2 = a2_0 * exp(a2_1 * vm)
    a3 = a3_0 * exp(a3_1 * vm)

    b1 = b1_0 * exp(-b1_1 * vm)
    b2 = b2_0 * exp(-b2_1 * vm)
    b3 = b3_0 * exp(-b3_1 * vm)
}
