INDEPENDENT { t FROM 0 TO 1 WITH 1 (ms) }

UNITS {
	(mV) = (millivolt)
	(mA) = (milliamp)
}
NEURON {
	SUFFIX test_channel
	USEION na READ ena WRITE ina
	RANGE gbar, ina, m, h, minf, mtau, hinf, htau
}
PARAMETER {
	gbar = 1.0 	   (mho/cm2)
}
ASSIGNED {
	v		(mV)
	ina 		(mA/cm2)
	ena 		(mV)
	minf 		(1)
	hinf 	   	(1)
	mtau		(ms)
	htau		(ms)

}
STATE {
	m h
}
BREAKPOINT {
	SOLVE states METHOD cnexp
	ina = gbar * m*m*m*h * (v - ena)
}
INITIAL {
	settables(v)
	m = minf
	h = hinf
}
DERIVATIVE states {
	settables(v)
	m' = (minf - m) / mtau
	h' = (hinf - h) / htau
}

UNITSOFF

PROCEDURE settables(vm(mV)) {
	TABLE minf, hinf, mtau, htau  FROM -100 TO 150 WITH 1000

	minf = 1/ (1+exp((vm-20)/5))
	mtau = 1
	hinf = 1/ (1+exp((-vm+50)/5))
	htau = 1
}

UNITSON
