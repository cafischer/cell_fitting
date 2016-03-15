TITLE Persistent sodium current

COMMENT 

animal: Long-Evans rats (Magistretti und Alonso, 1999), Sprague-Dawley 
rats (Huguenard et al., 1988)
cell type: stellate cell (Magistretti und Alonso, 1999), pyramidal cell (Huguenard et al., 1988)
region: Entorhinal Cortex layer 2 (Magistretti und Alonso, 1999), Neocortex (Huguenard et al., 1988)
current isolation: TTX current subtraction (Magistretti und Alonso, 1999), TTX (Huguenard et al., 1988) 
temperature: 22°C (Magistretti und Alonso, 1999), 23°C (Huguenard et al., 1988) 
temperature adjustment: activating component Q10 = 2.7, inactivating component Q10 = 1.3 (from guinea pig Purkinje cells!) (Kay et al., 1998)
model description: (Fransen et al., 2004)
experimental data: activation (fitted to Na current, not Nap!) (Huguenard et al., 1988) fitted by (McCormick & Huguenard, 1992), inactivation (Magistretti und Alonso, 1999)

references:
Fransén, E., Alonso, A. A., Dickson, C. T., Magistretti, J., & Hasselmo, M. E. (2004). Ionic mechanisms in the generation of subthreshold oscillations and action potential clustering in entorhinal layer II stellate neurons. Hippocampus, 14(3), 368-384.
Huguenard, J. R., Hamill, O. P., & Prince, D. A. (1988). Developmental changes in Na+ conductances in rat neocortical neurons: appearance of a slowly inactivating component. Journal of Neurophysiology, 59(3), 778-795.
Magistretti J, Alonso AA. 1999. Biophysical properties and slow voltage-
dependent inactivation of a sustained sodium current in entorhinal
cortex layer-II principal cells: a whole-cell and single-channel study.
J Gen Physiol 114:491–509.
McCormick, D. A., & Huguenard, J. R. (1992). A model of the electrophysiological properties of thalamocortical relay neurons. Journal of Neurophysiology, 68(4), 1384-1400.

link: https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=64296&file=/mitral/INaP.mod

author: -

ENDCOMMENT

INDEPENDENT { t FROM 0 TO 1 WITH 1 (ms) }

UNITS { 
	(mV) = (millivolt) 
	(mA) = (milliamp) 
} 
NEURON { 
	SUFFIX nap2
	USEION na READ ena WRITE ina
	RANGE gbar, ina2
}

PARAMETER { 
	gbar = 0.0003 	(mho/cm2)
	v 			(mV)
	q10_act = 2.7
	q10_inact = 1.3 
	celsius_channel = 22 	(degC) 
} 

ASSIGNED { 
	ina 		(mA/cm2) 
	ina2 		(mA/cm2) 
	ena 		(mV)
	minf 		(1)
	mtau 		(ms)
	hinf
	htau		(ms)
	tadj_act
	tadj_inact
	celsius (degC)
} 
STATE {
	m h
}

BREAKPOINT { 
	SOLVE states METHOD cnexp
	ina = 0
	ina2 = gbar * m * h * ( v - ena )
} 

INITIAL { 
	settables(v) 
	m = minf
	h = hinf
} 

DERIVATIVE states { 
	settables(v) 
	m' = ( minf - m ) / mtau
	h' = ( hinf - h ) / htau 
}
UNITSOFF
 
PROCEDURE settables(v) { 
	TABLE minf, mtau, hinf, htau DEPEND celsius FROM -120 TO 40 WITH 641

	tadj_act = q10_act^((celsius - celsius_channel) / 10 (degC))
	tadj_inact = q10_inact^((celsius - celsius_channel) / 10 (degC))	

	minf  = 1 / (1 + exp(-(v + 48.7) / 4.4 )) 
	if( v == -38.0 ) {
		mtau = .0013071895424837 :limit as v-->-38, a discontinuity in the mtau function
	}else{
		mtau = 1 / ((.091 * 1000 * (v + 38))/(1 - exp(-(v + 38)/5)) + (-.062 * 1000 * (v + 38))/(1 - exp((v + 38)/5)))
	}
	mtau = 1 / tadj_act * mtau
	hinf = 1 / ( 1 + exp((v + 48.8) / 9.98 ))
	htau = 1 / ((-2.88*.001*(v + 17.049))/(1 - exp((v - 49.1)/4.63)) + (6.94*.001*(v + 64.409))/(1 - exp(-(v + 447)/2.63)))
	htau = 1 / tadj_inact * htau
}
UNITSON
