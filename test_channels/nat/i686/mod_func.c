#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _hh2_reg(void);
extern void _nafast_reg(void);
extern void _narsg_reg(void);
extern void _nat_reg(void);
extern void _nax_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," hh2.mod");
    fprintf(stderr," nafast.mod");
    fprintf(stderr," narsg.mod");
    fprintf(stderr," nat.mod");
    fprintf(stderr," nax.mod");
    fprintf(stderr, "\n");
  }
  _hh2_reg();
  _nafast_reg();
  _narsg_reg();
  _nat_reg();
  _nax_reg();
}
