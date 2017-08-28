NEURON {
    SUFFIX kdr
    USEION k READ ek WRITE ik
    RANGE gbar, ik
    RANGE a1_0, a1_1, b1_0, b1_1, o_pow
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (S) = (siemens)
}

PARAMETER {
    gbar = 0.1     		 (S/cm2)

    o_pow = 4

    a1_0 = 10 (/ms)
    a1_1 = 0.001 (/mV)

    b1_0 = 0.001 (/ms)
    b1_1 = 0.1 (/mV)
}

ASSIGNED {
    v    (mV)
    ek  (mV)
    ik  (mA/cm2)
    a1   (/ms)
    b1   (/ms)
}

STATE {
    c o
}

BREAKPOINT {
    SOLVE kin METHOD sparse
    ik = gbar * pow(o, o_pow) * (v - ek)
}

INITIAL {
    SOLVE kin STEADYSTATE sparse
}

KINETIC kin {
    rates(v)
    ~ c <-> o (a1, b1)
    CONSERVE c + o = 1
}

PROCEDURE rates(vm (millivolt)) {
    a1 = a1_0 * exp(a1_1 * vm)

    b1 = b1_0 * exp(-b1_1 * vm)
}
