#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _IKM_reg(void);
extern void _Nap_reg(void);
extern void _hcn_reg(void);
extern void _hcn_fast_reg(void);
extern void _hcn_slow_reg(void);
extern void _kap_reg(void);
extern void _kdr_reg(void);
extern void _na8st_reg(void);
extern void _nat_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," IKM.mod");
    fprintf(stderr," Nap.mod");
    fprintf(stderr," hcn.mod");
    fprintf(stderr," hcn_fast.mod");
    fprintf(stderr," hcn_slow.mod");
    fprintf(stderr," kap.mod");
    fprintf(stderr," kdr.mod");
    fprintf(stderr," na8st.mod");
    fprintf(stderr," nat.mod");
    fprintf(stderr, "\n");
  }
  _IKM_reg();
  _Nap_reg();
  _hcn_reg();
  _hcn_fast_reg();
  _hcn_slow_reg();
  _kap_reg();
  _kdr_reg();
  _na8st_reg();
  _nat_reg();
}
