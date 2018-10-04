#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _k_hh_reg(void);
extern void _na_hh_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," k_hh.mod");
    fprintf(stderr," na_hh.mod");
    fprintf(stderr, "\n");
  }
  _k_hh_reg();
  _na_hh_reg();
}
