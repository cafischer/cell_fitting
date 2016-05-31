INDEPENDENT { t FROM 0 TO 1 WITH 1 (ms) }

UNITS {
	(mV) = (millivolt)
	(mA) = (milliamp)
}
NEURON {
	SUFFIX leak
	NONSPECIFIC_CURRENT i
	RANGE gbar, i, e
}
PARAMETER {
	gbar = 1.0 	   (mho/cm2)
    e = -70         (mV)
}
ASSIGNED {
	v		(mV)
	i 		(mA/cm2)

}
BREAKPOINT {
	i = gbar * (v - e)
}