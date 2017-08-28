UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
        (S) = (siemens)

}

NEURON {
        SUFFIX kdr
        USEION k READ ek WRITE ik
        RANGE gbar, ik
}

PARAMETER {
        gbar = 0.004         (S/cm2)

        shift = 10
        kShift = 10
        nTau = 50
}

STATE {
        n
}

ASSIGNED {
        ik       (mA/cm2)
        ek       (mV)
        ninf
        ntau     (ms)
}

INITIAL {
        rates(v)
        n = ninf
}        

BREAKPOINT {
        SOLVE states METHOD cnexp
        ik = gbar * n * (v - ek)
}

DERIVATIVE states {
        rates(v)
        n' = (ninf - n) / ntau
}

PROCEDURE rates(v (mV)) {
     ninf = 1 / (1 + exp(-0.12 * (v - 37 + kShift)))
     ntau = nTau * exp(-0.08 * (v - 37 + shift)) / (1 + exp(-0.12 * (v - 37 + kShift)))
}




















