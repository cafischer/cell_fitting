TITLE passive  (leak) membrane channel

COMMENT
from 
http://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=37857&file=/CN_Bushy_Stellate/leak.mod
ENDCOMMENT

UNITS {
	(mV) = (millivolt)
	(mA) = (milliamp)
}

NEURON {
	SUFFIX leak
	NONSPECIFIC_CURRENT i
	RANGE g, erev
}

PARAMETER {
	v (mV)
	g = 0.000058	(mho/cm2)
	erev = -83	(mV)
}

ASSIGNED { i	(mA/cm2)}

BREAKPOINT {
	i = g*(v - erev)
}








