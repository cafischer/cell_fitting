#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _cah_reg(void);
extern void _cat_reg(void);
extern void _kdr_reg(void);
extern void _km_reg(void);
extern void _nap_reg(void);
extern void _nat_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," cah.mod");
    fprintf(stderr," cat.mod");
    fprintf(stderr," kdr.mod");
    fprintf(stderr," km.mod");
    fprintf(stderr," nap.mod");
    fprintf(stderr," nat.mod");
    fprintf(stderr, "\n");
  }
  _cah_reg();
  _cat_reg();
  _kdr_reg();
  _km_reg();
  _nap_reg();
  _nat_reg();
}
