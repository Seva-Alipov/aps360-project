bool main(void){
  int iVar1;
  FILE *__stream;
  
  __stream = freopen("sample.txt","wb",stdout);
  iVar1 = fputs("ANY STRING TO WRITE TO A FILE AT ONCE.",__stream);
  return iVar1 < 1;
}
