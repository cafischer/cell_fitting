#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _hcn_fast_reg(void);
extern void _hcn_slow_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," hcn_fast.mod");
    fprintf(stderr," hcn_slow.mod");
    fprintf(stderr, "\n");
  }
  _hcn_fast_reg();
  _hcn_slow_reg();
}
