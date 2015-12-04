#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _CAtM95_reg(void);
extern void _ITGHK_reg(void);
extern void _cad_reg(void);
extern void _cal2_reg(void);
extern void _cat_reg(void);
extern void _ical_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," CAtM95.mod");
    fprintf(stderr," ITGHK.mod");
    fprintf(stderr," cad.mod");
    fprintf(stderr," cal2.mod");
    fprintf(stderr," cat.mod");
    fprintf(stderr," ical.mod");
    fprintf(stderr, "\n");
  }
  _CAtM95_reg();
  _ITGHK_reg();
  _cad_reg();
  _cal2_reg();
  _cat_reg();
  _ical_reg();
}
