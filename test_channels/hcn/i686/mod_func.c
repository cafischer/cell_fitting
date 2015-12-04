#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _hcn_reg(void);
extern void _leak_reg(void);
extern void _nap_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," hcn.mod");
    fprintf(stderr," leak.mod");
    fprintf(stderr," nap.mod");
    fprintf(stderr, "\n");
  }
  _hcn_reg();
  _leak_reg();
  _nap_reg();
}
