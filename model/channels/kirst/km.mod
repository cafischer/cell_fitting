UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
        SUFFIX km
        USEION k READ ek WRITE ik
        RANGE gbar, ik
        }

PARAMETER {
        gbar = 0 (S/cm2)

		mShift = 20
		mMTau = 1.0
}

STATE {
        n
}

ASSIGNED {
        v (mV)
        ek (mV)
        ik (mA/cm2)
        mMinf
	    mMtau (ms)
}

BREAKPOINT {
        SOLVE states METHOD cnexp
	    ik = gbar * n * (v - ek)
}


INITIAL {
	rates(v)
	n = mMinf
}

DERIVATIVE states {
        rates(v)
        n' =  (mMinf-n) / mMtau
}


PROCEDURE rates(v(mV)) {

UNITSOFF

      mMinf = 1 / (1 + exp(-0.1 * (v + 16 + mShift)))
      mMtau = mMTau * (60 + 333 * exp(-0.106 * (v + 18 + mShift)) / (1 + exp(-0.265 * (v + 18 + mShift))))

UNITSON
}