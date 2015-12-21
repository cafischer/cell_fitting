TITLE leak potassium current

COMMENT

animal: mice (Garden, 2008)
cell type: stellate cell (Garden, 2008)
region: Entorhinal Cortex layer 2 (Garden, 2008)
current isolation: quinidine current subtraction (Garden, 2008)
temperature: -
temperature adjustment: -
model description: -
experimental data: (Garden, 2008)

references:
Garden, D. L., Dodson, P. D., O'Donnell, C., White, M. D., & Nolan, M. F. (2008). Tuning of synaptic integration in the medial entorhinal cortex to the organization of grid cell firing fields. Neuron, 60(5), 875-889.

link: http://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=37857&file=/CN_Bushy_Stellate/leak.mod

author: Neuron Book

ENDCOMMENT

UNITS {
	(mV) = (millivolt)
	(mA) = (milliamp)
}

NEURON {
	SUFFIX kleak
	NONSPECIFIC_CURRENT i
	RANGE i, gbar, ekleak
}

PARAMETER {
	gbar = 0.0001	(mho/cm2) 
	ekleak = -100	(mV)  
}

ASSIGNED { 
	i	(mA/cm2)
	v 	(mV)
}

BREAKPOINT {
	i = gbar * (v - ekleak)
}








