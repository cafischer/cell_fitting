#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _cav_reg(void);
extern void _hcn_fast_reg(void);
extern void _hcn_slow_reg(void);
extern void _ka_reg(void);
extern void _ka2_reg(void);
extern void _kdr_reg(void);
extern void _kdr2_reg(void);
extern void _kdri_reg(void);
extern void _nap_reg(void);
extern void _nap2_reg(void);
extern void _nap_act_reg(void);
extern void _nap_act2_reg(void);
extern void _nap_act_fast_reg(void);
extern void _nap_act_slow_reg(void);
extern void _nap_inact_reg(void);
extern void _nap_markov_reg(void);
extern void _nat_reg(void);
extern void _nat_act_reg(void);
extern void _nat_ep_reg(void);
extern void _nats_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," cav.mod");
    fprintf(stderr," hcn_fast.mod");
    fprintf(stderr," hcn_slow.mod");
    fprintf(stderr," ka.mod");
    fprintf(stderr," ka2.mod");
    fprintf(stderr," kdr.mod");
    fprintf(stderr," kdr2.mod");
    fprintf(stderr," kdri.mod");
    fprintf(stderr," nap.mod");
    fprintf(stderr," nap2.mod");
    fprintf(stderr," nap_act.mod");
    fprintf(stderr," nap_act2.mod");
    fprintf(stderr," nap_act_fast.mod");
    fprintf(stderr," nap_act_slow.mod");
    fprintf(stderr," nap_inact.mod");
    fprintf(stderr," nap_markov.mod");
    fprintf(stderr," nat.mod");
    fprintf(stderr," nat_act.mod");
    fprintf(stderr," nat_ep.mod");
    fprintf(stderr," nats.mod");
    fprintf(stderr, "\n");
  }
  _cav_reg();
  _hcn_fast_reg();
  _hcn_slow_reg();
  _ka_reg();
  _ka2_reg();
  _kdr_reg();
  _kdr2_reg();
  _kdri_reg();
  _nap_reg();
  _nap2_reg();
  _nap_act_reg();
  _nap_act2_reg();
  _nap_act_fast_reg();
  _nap_act_slow_reg();
  _nap_inact_reg();
  _nap_markov_reg();
  _nat_reg();
  _nat_act_reg();
  _nat_ep_reg();
  _nats_reg();
}
