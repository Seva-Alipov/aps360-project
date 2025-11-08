
#define _CRT_SECURE_NO_WARNINGS  
#include <stdio.h>

int main(void)
{
    return 0 >= fputs("ANY STRING TO WRITE TO A FILE AT ONCE.", 
        freopen("sample.txt","wb",stdout));
}
