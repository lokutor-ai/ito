#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ito_espeak.h"
int main(int argc, char **argv){
  FILE *f=fopen(argv[1],"rb"); fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); char *b=malloc(n); fread(b,1,n,f);
  int r=itoes_init(b,n); if(r){printf("init %d\n",r);return 1;}
  char out[8192];
  for(int i=2;i<argc;i++){ int k=itoes_text_to_phonemes(argv[i],out,sizeof out); printf("%d [%s]\n",k,out);} return 0;}
