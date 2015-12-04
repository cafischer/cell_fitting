#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _BK_reg(void);
extern void _IAHP_reg(void);
extern void _SK_reg(void);
extern void _bkkca_reg(void);
extern void _cad_reg(void);
extern void _kc_reg(void);
extern void _kca_reg(void);
extern void _kca3_reg(void);
extern void _skkca_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," BK.mod");
    fprintf(stderr," IAHP.mod");
    fprintf(stderr," SK.mod");
    fprintf(stderr," bkkca.mod");
    fprintf(stderr," cad.mod");
    fprintf(stderr," kc.mod");
    fprintf(stderr," kca.mod");
    fprintf(stderr," kca3.mod");
    fprintf(stderr," skkca.mod");
    fprintf(stderr, "\n");
  }
  _BK_reg();
  _IAHP_reg();
  _SK_reg();
  _bkkca_reg();
  _cad_reg();
  _kc_reg();
  _kca_reg();
  _kca3_reg();
  _skkca_reg();
}
