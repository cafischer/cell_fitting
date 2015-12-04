#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _na8st_reg(void);
extern void _narsg_reg(void);
extern void _nat_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," na8st.mod");
    fprintf(stderr," narsg.mod");
    fprintf(stderr," nat.mod");
    fprintf(stderr, "\n");
  }
  _na8st_reg();
  _narsg_reg();
  _nat_reg();
}
