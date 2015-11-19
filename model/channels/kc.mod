TITLE Potassium C type current for RD Traub, J Neurophysiol 89:909-921, 2003

COMMENT

temperature adjustment: Q10 = 1.6 (Brown, 1983)

references:
Brown, D. A., & Griffith, W. H. (1983). Calcium‐activated outward current in voltage‐clamped hippocampal neurones of the guinea‐pig. The Journal of physiology, 337(1), 287-301.

link: http://senselab.med.yale.edu/modeldb/showmodel.cshtml?model=20756&file=\traub2003\kc.mod

author: Maciej Lazarewicz (mlazarew@seas.upenn.edu)

ENDCOMMENT

INDEPENDENT { t FROM 0 TO 1 WITH 1 (ms) }

UNITS { 
	(mV) = (millivolt) 
	(mA) = (milliamp) 
}
 
NEURON { 
	SUFFIX kc
	USEION k READ ek WRITE ik
	:USEION ca READ cai  TODO
	RANGE  gbar, ik
}

PARAMETER { 
	gbar = 0.0 	(mho/cm2)
	v ek 		(mV)  
	cai = 2		(mM)

	temp = 20       (degC)
	q10 = 1.6	
} 

ASSIGNED { 
	ik 		(mA/cm2) 
	alpha beta	(/ms)
	tadj
}
 
STATE {
	m
}

BREAKPOINT { 
	SOLVE states METHOD cnexp
	if( 0.004 * cai < 1 ) {
		ik = gbar * m * 0.004 * cai * ( v - ek ) 
	}else{
		ik = gbar * m * ( v - ek ) 
	}
}
 
INITIAL { 
	settables(v) 
	m = alpha / ( alpha + beta )
	m = 0
}
 
DERIVATIVE states { 
	settables(v) 
	m' = alpha * ( 1 - m ) - beta * m 
}

UNITSOFF 

PROCEDURE settables(v) { 
	TABLE alpha, beta 
	DEPEND celsius, temp
	FROM -120 TO 40 WITH 641

	tadj = q10^((celsius - temp)/10 (degC))

	if( v < -10.0 ) {
		alpha = (2 / 37.95 * (exp((v + 50) / 11 - (v + 53.5) / 27))) / tadj
	: Note typo in the paper: missing minus sign in the front of 'v'
		beta  = (2 * exp(( - v - 53.5) / 27) - alpha) / tadj
	}else{
		alpha = (2 * exp(( - v - 53.5) / 27 )) / tadj
		beta  = 0
	}
}

UNITSON





