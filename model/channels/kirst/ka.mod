UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
        SUFFIX ka
        USEION k READ ek WRITE ik
        RANGE gbar, ik, n, l
        }

PARAMETER {
        gbar = 0.055 (S/cm2)

		aShift = 10
		nATau = 3.4
		lATau = 1
		maxk = 40
}

STATE {
        n
        l
}

ASSIGNED {
        v (mV)
        ek (mV)
        ik (mA/cm2)
        nAinf
	    nAtau (ms)
        lAinf
	    lAtau (ms)
}

BREAKPOINT {
        SOLVE states METHOD cnexp
	    ik = gbar * n * l * (v - ek)
}


INITIAL {
	rates(v)
	n = nAinf
    l = lAinf
}

DERIVATIVE states {
        rates(v)
        n' =  (nAinf - n) / nAtau
        l' =  (lAinf - l) / lAtau
}


PROCEDURE rates(v(mV)) {
	LOCAL z

UNITSOFF
      z = -1.5 - 1/(1+exp((v + 16 + aShift)/5))

	  nAinf = 1 / (1 + exp(0.038 * z * (v - 35 + aShift)))
      nAtau = nATau * exp(0.021 * z * (v - 35 + aShift)) / (1 + exp(0.038 * z * (v - 35 + aShift)))

      lAinf = 1 / (1 + exp(0.12 * (v + 32 + aShift)))
      lAtau = lATau * SoftMax(0.1, 0.26 * (v + 26 + aShift))

UNITSON
}

FUNCTION SoftMax(a, b) {
		SoftMax = log(exp(maxk*a) + exp(maxk*b)) / maxk
}