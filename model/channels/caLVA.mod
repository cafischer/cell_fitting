TITLE Low threshold calcium current

COMMENT

animal: Sprague-Dawley rats (Huguenard, 1992a)
cell type: thalamic relay neurons (Huguenard, 1992a)
region: thalamus (Huguenard, 1992a)
current isolation: tail current (Huguenard, 1992a) 
temperature: 23-25°C (Huguenard, 1992a)
temperature adjustment: activating component Q10 = 5, inactivating component Q10 = 3 (Coulter, 1989)
model description: (Huguenard, 1992b), (Destexhe, 1998)
experimental data: (Huguenard, 1992a)

references:
Huguenard, J. R., & Prince, D. A. (1992a). A novel T-type current underlies prolonged Ca (2+)-dependent burst firing in GABAergic neurons of rat thalamic reticular nucleus. The Journal of neuroscience, 12(10), 3804-3817.
Huguenard, J. R., & McCormick, D. A. (1992b). Simulation of the currents involved in rhythmic oscillations in thalamic relay neurons. Journal of Neurophysiology, 68(4), 1373-1383.
Coulter, D. A., Huguenard, J. R., & Prince, D. A. (1989). Calcium currents in rat thalamocortical relay neurones: kinetic properties of the transient, low‐threshold current. The Journal of physiology, 414(1), 587-604.
Destexhe, A., Neubig, M., Ulrich, D., & Huguenard, J. (1998). Dendritic low-threshold calcium currents in thalamic relay cells. The Journal of neuroscience, 18(10), 3574-3588. (http://cns.fmed.ulaval.ca)

link: http://senselab.med.yale.edu/modeldb/showmodel.cshtml?model=279&file=\dendtc\itghk.mod

author: Alain Destexhe

additional comments: 
- shift parameter for screening charge
- empirical correction for contamination by inactivation (Huguenard)
- GHK equations
ENDCOMMENT

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX caLVA
	USEION ca READ cai,cao WRITE ica
	RANGE pbar, m_inf, tau_m, h_inf, tau_h, shift, actshift, ica
	GLOBAL qm, qh
}

UNITS {
	(molar) = (1/liter)
	(mV) =	(millivolt)
	(mA) =	(milliamp)
	(mM) =	(millimolar)

	FARADAY = (faraday) (coulomb)
	R = (k-mole) (joule/degC)
}

PARAMETER {
	v		(mV)
	pbar = .2e-3    (cm/s) 	  : max. permeability
	shift	=  2	(mV)	  : corresponds to 2mM ext Ca++ 
	actshift = 0 	(mV)	  : shift of activation curve (towards hyperpol)
	qm	= 5		  : q10's for activation and inactivation
	qh	= 3		  : from Coulter et al., J Physiol 414: 587, 1989
}

STATE {
	m h
}

ASSIGNED {
	ica	(mA/cm2)
	cai     (Mm)
	cao	(Mm)
	m_inf
	tau_m	(ms)
	h_inf
	tau_h	(ms)
	phi_m
	phi_h
	celsius (degC)
}

BREAKPOINT {
	SOLVE castate METHOD cnexp
	ica = pbar * m*m*h * ghk(v, cai, cao)
}

DERIVATIVE castate {
	evaluate_fct(v)

	m' = (m_inf - m) / tau_m
	h' = (h_inf - h) / tau_h
}


UNITSOFF
INITIAL {
	phi_m = qm ^ ((celsius-24)/10)
	phi_h = qh ^ ((celsius-24)/10)

	evaluate_fct(v)

	m = m_inf
	h = h_inf
}

PROCEDURE evaluate_fct(v(mV)) {
	m_inf = 1.0 / ( 1 + exp(-(v+shift+actshift+57)/6.2) )
	h_inf = 1.0 / ( 1 + exp((v+shift+81)/4.0) )

	tau_m = ( 0.612 + 1.0 / ( exp(-(v+shift+actshift+132)/16.7) + exp((v+shift+actshift+16.8)/18.2) ) ) / phi_m
	if( (v+shift) < -80) {
		tau_h = exp((v+shift+467)/66.6) / phi_h
	} else {
		tau_h = ( 28 + exp(-(v+shift+22)/10.5) ) / phi_h
	}
}

FUNCTION ghk(v(mV), ci(mM), co(mM)) (.001 coul/cm3) {
	LOCAL z, eci, eco
	z = (1e-3)*2*FARADAY*v/(R*(celsius+273.15))
	eco = co*efun(z)
	eci = ci*efun(-z)
	:high cao charge moves inward
	:negative potential charge moves inward
	ghk = (.001)*2*FARADAY*(eci - eco)
}

FUNCTION efun(z) {
	if (fabs(z) < 1e-4) {
		efun = 1 - z/2
	}else{
		efun = z/(exp(z) - 1)
	}
}
FUNCTION nongat(v,cai,cao) {	: non gated current
	nongat = pbar * ghk(v, cai, cao)
}
UNITSON






