TITLE hh.mod   squid sodium, potassium, and leak channels
 
COMMENT
 This is the original Hodgkin-Huxley treatment for the set of sodium, 
  potassium, and leakage channels found in the squid giant axon membrane.
  ("A quantitative description of membrane current and its application 
  conduction and excitation in nerve" J.Physiol. (Lond.) 117:500-544 (1952).)
 Membrane voltage is in absolute mV and has been reversed in polarity
  from the original HH convention and shifted to reflect a resting potential
  of -65 mV.
 Remember to set celsius=6.3 (or whatever) in your HOC file.
 See squid.hoc for an example of a simulation using this model.
 SW Jaslove  6 March, 1992
ENDCOMMENT
 
UNITS {
        (mA) = (milliamp)
        (mV) = (millivolt)
	(S) = (siemens)
}
 
? interface
NEURON {
        SUFFIX k_hh
        USEION k READ ek WRITE ik
        RANGE gbar, ik
        GLOBAL ninf, ntau
	THREADSAFE 
}
 
PARAMETER {
        gbar = .036 (S/cm2)	<0,1e9>
}
 
STATE {
        n
}
 
ASSIGNED {
        v (mV)
        celsius (degC)
        ek (mV)

	gk (S/cm2)
        ik (mA/cm2)
	ninf
	ntau (ms)
}
 
? currents
BREAKPOINT {
        SOLVE states METHOD cnexp
        gk = gbar*n*n*n*n
	ik = gk*(v - ek)      
}
 
 
INITIAL {
	rates(v)
	n = ninf
}

? states
DERIVATIVE states {  
        rates(v)
        n' = (ninf-n)/ntau
}


? rates
PROCEDURE rates(v(mV)) {  
        LOCAL  alpha, beta, sum, q10
        TABLE ninf, ntau DEPEND celsius FROM -100 TO 100 WITH 200

UNITSOFF
        q10 = 3^((celsius - 6.3)/10)
                :"n" potassium activation system
        alpha = .01*vtrap(-(v+55),10) 
        beta = .125*exp(-(v+65)/80)
	sum = alpha + beta
        ntau = 1/(q10*sum)
        ninf = alpha/sum
}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{
                vtrap = x/(exp(x/y) - 1)
        }
}
 
UNITSON
