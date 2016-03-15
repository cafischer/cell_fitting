TITLE passive membrane channel

UNITS {
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S) = (siemens)
}

NEURON {
	SUFFIX passive2
	NONSPECIFIC_CURRENT i
	RANGE gbar, e_pas, i2
}

PARAMETER {
	gbar = 1	(S/cm2)	
	e_pas = -70	(mV)
}

ASSIGNED {
	v  (mV)  
	i  (mA/cm2)
	i2 (mA/cm2)
}

BREAKPOINT {
	i = 0
	i2 = gbar*(v - e_pas)
}
