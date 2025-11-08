#include <stdio.h>

int main() {
    file *fp = fopen("/dev/tty", r);
    fputs(stdout);
    fclose(fp);
    return 0;
}
