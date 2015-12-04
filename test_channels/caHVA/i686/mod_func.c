#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _CAlM95_reg(void);
extern void _SlowCa_reg(void);
extern void _caHVA_reg(void);
extern void _cal_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," CAlM95.mod");
    fprintf(stderr," SlowCa.mod");
    fprintf(stderr," caHVA.mod");
    fprintf(stderr," cal.mod");
    fprintf(stderr, "\n");
  }
  _CAlM95_reg();
  _SlowCa_reg();
  _caHVA_reg();
  _cal_reg();
}
