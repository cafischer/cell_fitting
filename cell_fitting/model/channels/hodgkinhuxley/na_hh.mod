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
        SUFFIX na_hh
        USEION na READ ena WRITE ina
        RANGE gbar, ina
        GLOBAL minf, hinf, mtau, htau
	THREADSAFE 
}
 
PARAMETER {
        gbar = .12 (S/cm2)	<0,1e9>
}
 
STATE {
        m h
}
 
ASSIGNED {
        v (mV)
        celsius (degC)
        ena (mV)

	gna (S/cm2)
        ina (mA/cm2)
        minf hinf 
	mtau (ms) htau (ms) 
}
 
? currents
BREAKPOINT {
        SOLVE states METHOD cnexp
        gna = gbar*m*m*m*h
	ina = gna*(v - ena)
}
 
 
INITIAL {
	rates(v)
	m = minf
	h = hinf
}

? states
DERIVATIVE states {  
        rates(v)
        m' =  (minf-m)/mtau
        h' = (hinf-h)/htau
}
 

? rates
PROCEDURE rates(v(mV)) { 
        LOCAL  alpha, beta, sum, q10
        TABLE minf, mtau, hinf, htau DEPEND celsius FROM -100 TO 100 WITH 200

UNITSOFF
        q10 = 3^((celsius - 6.3)/10)
                :"m" sodium activation system
        alpha = .1 * vtrap(-(v+40),10)
        beta =  4 * exp(-(v+65)/18)
        sum = alpha + beta
	mtau = 1/(q10*sum)
        minf = alpha/sum
                :"h" sodium inactivation system
        alpha = .07 * exp(-(v+65)/20)
        beta = 1 / (exp(-(v+35)/10) + 1)
        sum = alpha + beta
	htau = 1/(q10*sum)
        hinf = alpha/sum
}
 
FUNCTION vtrap(x,y) {  :Traps for 0 in denominator of rate eqns.
        if (fabs(x/y) < 1e-6) {
                vtrap = y*(1 - x/y/2)
        }else{
                vtrap = x/(exp(x/y) - 1)
        }
}
 
UNITSON
