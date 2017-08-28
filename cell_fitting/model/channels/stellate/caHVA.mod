TITLE high-voltage activated calcium current

COMMENT
animal: 
cell type: 
region: 
current isolation:  
temperature: 24°C
temperature adjustment: activating component Q10 = 5.9, inactivating component Q10 = 2.0 (from chick sensory neurons) (Acerbo, 1994)
model description: 
experimental data: (Reuveni, 1993)

references:
Reuveni, I., Friedman, A., Amitai, Y., & Gutnick, M. J. (1993). Stepwise repolarization from Ca2+ plateaus in neocortical pyramidal cells: evidence for nonhomogeneous distribution of HVA Ca2+ channels in dendrites. The Journal of neuroscience, 13(11), 4609-4621.
Acerbo, P., & Nobile, M. (1994). Temperature dependence of multiple high voltage activated Ca2+ channels in chick sensory neurones. European biophysics journal, 23(3), 189-195.

link: http://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=2488&file=/patdemo/ca.mod

author: Zach Mainen, zach@salk.edu

additional comment:
Uses fixed eca instead of GHK eqn
ENDCOMMENT

NEURON {
    THREADSAFE
	SUFFIX caHVA
	USEION ca READ eca WRITE ica
	RANGE m, h, gca, gbar, ica
	RANGE minf, hinf, mtau, htau
	GLOBAL q10_act, q10_inact, temp, tadj_act, tadj_inact, vmin, vmax, vshift
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
    (pS) = (picosiemens)
    (um) = (micron)
    (mM) = (milli/liter)
    FARADAY = (faraday) (coulomb)
    R = (k-mole) (joule/degC)
    PI	= (pi) (1)
} 

PARAMETER {
	gbar = 0.1   	(pS/um2)	: 0.12 mho/cm2
	vshift = 0	(mV)		: voltage shift (affects all)
	temp = 23	(degC)		: original temp 
	q10_act = 5.9
	q10_inact = 2.0
	vmin = -120	(mV)
	vmax = 100	(mV)
}

ASSIGNED {
    v 		(mV)
    celsius	(degC)
    ica 	(mA/cm2)
    gca		(pS/um2)
    eca		(mV)
    minf 		
    hinf
    mtau 	(ms)
    htau 	(ms)
    tadj_act
    tadj_inact
}
 
STATE { m h }

INITIAL {
    trates(v+vshift)
    m = minf
    h = hinf
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    gca = gbar*m*m*h   
    ica = (1e-4) * gca * (v - eca)
} 

DERIVATIVE states {
        trates(v+vshift)      
        m' =  (minf-m)/mtau
        h' =  (hinf-h)/htau
}

PROCEDURE trates(v (mV)) {  
    TABLE minf, hinf, mtau, htau
    DEPEND celsius, temp
    FROM vmin TO vmax WITH 199

    rates(v): not consistently executed from here if usetable == 1
}

UNITSOFF
PROCEDURE rates(vm (mV)) {  
    LOCAL  a, b

    tadj_act = q10_act^((celsius - temp)/(10 (degC)))
    tadj_inact = q10_inact^((celsius - temp)/(10 (degC)))

    : m: activation
    a = 0.209*efun(-(27+vm)/3.8)
    b = 0.94*exp((-75-vm)/17)
	
    mtau = 1/tadj_act/(a+b)
    minf = a/(a+b)

    : h: inactivation 
    a = 0.000457*exp((-13-vm)/50)
    b = 0.0065/(exp((-vm-15)/28) + 1)

    htau = 1/tadj_inact/(a+b)
    hinf = a/(a+b)
}
UNITSON

FUNCTION efun(z) {
	if (fabs(z) < 1e-4) {
		efun = 1 - z/2
	}else{
		efun = z/(exp(z) - 1)
	}
}






