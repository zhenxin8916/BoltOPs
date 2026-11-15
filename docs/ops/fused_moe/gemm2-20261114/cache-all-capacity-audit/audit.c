#define _GNU_SOURCE
#include <link.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
static void (*real_tag)(void*,int,int,int,int);
static void (*real_tcc)(void*,const void*);
static __thread int in_tcc;
static __thread int in_tcp;
static void (*real_tcp)(void*,const void*);
static void tag_wrapper(void *p,int a,int b,int c,int d) {
 if(in_tcp) fprintf(stderr,"TCP_TAGSTORE_CONSTRUCTOR args=%d,%d,%d,%d\n",a,b,c,d);
 if(in_tcc) fprintf(stderr,"TCC_TAGSTORE_CONSTRUCTOR args=%d,%d,%d,%d\n",a,b,c,d);
 real_tag(p,a,b,c,d);
}
static void tcc_wrapper(void*p,const void*q) {in_tcc++;real_tcc(p,q);in_tcc--;}
static void tcp_wrapper(void*p,const void*q) {in_tcp++;real_tcp(p,q);in_tcp--;}
unsigned int la_version(unsigned int v) {return LAV_CURRENT;}
unsigned int la_objopen(struct link_map*m,Lmid_t l,uintptr_t*c) {return LA_FLG_BINDTO|LA_FLG_BINDFROM;}
uintptr_t la_symbind64(Elf64_Sym*s,unsigned int n,uintptr_t*r,uintptr_t*d,unsigned int*f,const char*name) {
 if(!strcmp(name,"_ZN8TagStoreC1Eiiii") || !strcmp(name,"_ZN8TagStoreC2Eiiii")) {real_tag=(void*)s->st_value; return (uintptr_t)tag_wrapper;}
 if(!strcmp(name,"_ZN3TCCC1ERK9TCCParams") || !strcmp(name,"_ZN3TCCC2ERK9TCCParams")) {real_tcc=(void*)s->st_value; return (uintptr_t)tcc_wrapper;}
 if(!strcmp(name,"_ZN3TCPC1ERK9TCPParams") || !strcmp(name,"_ZN3TCPC2ERK9TCPParams")) {real_tcp=(void*)s->st_value; return (uintptr_t)tcp_wrapper;}
 return s->st_value;
}
