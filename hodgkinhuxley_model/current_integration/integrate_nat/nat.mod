TITLE Transient sodium current

COMMENT

animal: Sprague-Dawley rats (Huguenard et al., 1988)
cell type: pyramidal cell (Huguenard et al., 1988)
region: Neocortex (Huguenard et al., 1988)
current isolation: TTX current subtraction (Huguenard et al., 1988) 
temperature: 23°C (Huguenard et al., 1988) 
temperature adjustment: Q10 = 3 (Hodgkin, 1952)
model description: - 
experimental data: (Huguenard et al., 1988), (Hamill et al., 1991)

references:
Huguenard, J. R., Hamill, O. P., & Prince, D. A. (1988). Developmental changes in Na+ conductances in rat neocortical neurons: appearance of a slowly inactivating component. Journal of Neurophysiology, 59(3), 778-795.
Hamill, O. P., Huguenard, J. R., & Prince, D. A. (1991). Patch-clamp studies of voltage-gated currents in identified neurons of the rat cerebral cortex. Cerebral Cortex, 1(1), 48-61.
Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane current and its application to conduction and excitation in nerve. The Journal of physiology, 117(4), 500-544.

link: https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=2488&file=/patdemo/na.mod

author: Zach Mainen (zach@salk.edu)

additional comments:
qi is not well constrained by the data, since there are no points
between -80 and -55.
Voltage dependencies are shifted approximately from the best
fit to give higher threshold.
ENDCOMMENT

NEURON {
    THREADSAFE
	SUFFIX nat
	USEION na READ ena WRITE ina
	RANGE m, h, gna, gbar, ina
	GLOBAL tha, thi1, thi2, qa, qi, qinf, thinf
	RANGE minf, hinf, mtau, htau
	GLOBAL Ra, Rb, Rd, Rg
	GLOBAL vmin, vmax
	RANGE vshift
}

UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)
	(pS) = (picosiemens)
	(um) = (micron)
} 

PARAMETER {
	gbar = 10       (pS/um2)
								
	tha  = -35	(mV)		: v 1/2 for act
	qa   = 9	(mV)		: act slope		
	Ra   = 0.182	(/ms)		: open (v)		
	Rb   = 0.124	(/ms)		: close (v)		

	thi1  = -50	(mV)		: v 1/2 for inact 	
	thi2  = -75	(mV)		: v 1/2 for inact 	
	qi   = 5	(mV)	        : inact tau slope
	thinf  = -65	(mV)		: inact inf slope	
	qinf  = 6.2	(mV)		: inact inf slope
	Rg   = 0.0091	(/ms)		: inact (v)	
	Rd   = 0.024	(/ms)		: inact recov (v)

	vmin = -120	(mV)
	vmax = 100	(mV)
}

ASSIGNED {
	v 		(mV)
	ina 		(mA/cm2)
	gna		(pS/um2)
	ena		(mV)
	minf 		
	hinf
	mtau (ms)	
	htau (ms)
}
 
STATE { m h }

INITIAL {
	trates(v)
	m = minf
	h = hinf
}

BREAKPOINT {
        SOLVE states METHOD cnexp
        gna = gbar*m*m*m*h
	ina = (1e-4) * gna * (v - ena)
} 

DERIVATIVE states {  
        trates(v)
        m' =  (minf-m)/mtau
        h' =  (hinf-h)/htau
}

PROCEDURE trates(v (mV)) {  
    TABLE minf,  hinf, mtau, htau
    DEPEND Ra, Rb, Rd, Rg, tha, thi1, thi2, qa, qi, qinf
    FROM vmin TO vmax WITH 1000

    rates(v)	: not consistently executed from here if usetable == 1
}

UNITSOFF
PROCEDURE rates(vm (mV)) {  
    LOCAL  a, b

    a = Ra * qa * efun((tha - vm)/qa)
    b = Rb * qa * efun((vm - tha)/qa)

    mtau = 1/(a+b)
    minf = a/(a+b)

    a = Rd * qi * efun((thi1 - vm)/qi)
    b = Rg * qi * efun((vm - thi2)/qi)

    htau = 1/(a+b)
    hinf = 1/(1+exp((vm-thinf)/qinf))
}
UNITSON

FUNCTION efun(z) {
	if (fabs(z) < 1e-6) {
		efun = 1 - z/2
	}else{
		efun = z/(exp(z) - 1)
	}
}






