#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _IKM_reg(void);
extern void _Nap_schmidthieber_reg(void);
extern void _caHVA_reg(void);
extern void _caLVA_reg(void);
extern void _cad_reg(void);
extern void _hcn_reg(void);
extern void _ka_reg(void);
extern void _kca_reg(void);
extern void _kdr_reg(void);
extern void _kleak_reg(void);
extern void _na8st_reg(void);
extern void _nap_reg(void);
extern void _narsg_reg(void);
extern void _nat_reg(void);
extern void _nav16_reg(void);
extern void _passive_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," IKM.mod");
    fprintf(stderr," Nap_schmidthieber.mod");
    fprintf(stderr," caHVA.mod");
    fprintf(stderr," caLVA.mod");
    fprintf(stderr," cad.mod");
    fprintf(stderr," hcn.mod");
    fprintf(stderr," ka.mod");
    fprintf(stderr," kca.mod");
    fprintf(stderr," kdr.mod");
    fprintf(stderr," kleak.mod");
    fprintf(stderr," na8st.mod");
    fprintf(stderr," nap.mod");
    fprintf(stderr," narsg.mod");
    fprintf(stderr," nat.mod");
    fprintf(stderr," nav16.mod");
    fprintf(stderr," passive.mod");
    fprintf(stderr, "\n");
  }
  _IKM_reg();
  _Nap_schmidthieber_reg();
  _caHVA_reg();
  _caLVA_reg();
  _cad_reg();
  _hcn_reg();
  _ka_reg();
  _kca_reg();
  _kdr_reg();
  _kleak_reg();
  _na8st_reg();
  _nap_reg();
  _narsg_reg();
  _nat_reg();
  _nav16_reg();
  _passive_reg();
}
