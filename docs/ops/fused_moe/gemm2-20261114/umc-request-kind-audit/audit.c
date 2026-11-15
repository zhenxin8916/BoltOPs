#define _GNU_SOURCE
#include <link.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
static void* (*real_get)(void*);
static void* get_wrapper(void *p) {
 void *pkt=real_get(p);
 if(pkt) {
  unsigned cmd; uint64_t id;
  memcpy(&cmd,(char*)pkt+0xc,sizeof(cmd));
  memcpy(&id,(char*)pkt+0x10,sizeof(id));
  fprintf(stderr,"UMC_REQUEST cmd=%u id=%lu\n",cmd,(unsigned long)id);
 }
 return pkt;
}
unsigned int la_version(unsigned int v) {return LAV_CURRENT;}
unsigned int la_objopen(struct link_map*m,Lmid_t l,uintptr_t*c) {return LA_FLG_BINDTO|LA_FLG_BINDFROM;}
uintptr_t la_symbind64(Elf64_Sym*s,unsigned int n,uintptr_t*r,uintptr_t*d,unsigned int*f,const char*name) {
 if(!strcmp(name,"_ZN15MemSimulateBase8RespPort9getReqPktEv")) {real_get=(void*(*)(void*))s->st_value; return (uintptr_t)get_wrapper;}
 return s->st_value;
}
