UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	    (S) = (siemens)
}

NEURON {
        SUFFIX nat
        USEION na READ ena WRITE ina
        RANGE gbar, ina, m, h
        }

PARAMETER {
        gbar = 0.05 (S/cm2)
		shift = 10
		hShift = 10
		mTau = 0.5
		hTau = 3
}

STATE {
        m h
}

ASSIGNED {
        v (mV)
        ena (mV)
        ina (mA/cm2)
        minf
        hinf
	    mtau (ms)
        htau (ms)
}

BREAKPOINT {
        SOLVE states METHOD cnexp
	    ina = gbar*m*m*m*h*(v - ena)
}


INITIAL {
	rates(v)
	m = minf
	h = hinf
}

DERIVATIVE states {
        rates(v)
        m' =  (minf-m) / mtau
        h' = (hinf-h) / htau
}


PROCEDURE rates(v(mV)) {
LOCAL am, bm, ah, bh

UNITSOFF
	  am = 0.4 * (v + 6 + shift) / (1 - exp(-(v + 6 + shift)/7.2))
      bm = -0.124 * (v + 6 + shift) / (1 - exp((v + 6 + shift)/7.2))
      ah = 0.03  * (v + 21 + hShift) / (1 - exp(-(v + 21 + hShift)/1.5))
      bh = -0.01 * (v + 21 + hShift) / (1 - exp((v + 21 + hShift) / 1.5))

      minf = am / (am + bm)
      hinf = 1/ (1 + exp((v + 26 + hShift)/4))

      mtau = mTau / (am + bm)
      htau = hTau / (ah + bh)
UNITSON
}