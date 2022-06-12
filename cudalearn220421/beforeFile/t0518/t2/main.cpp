// main.cpp
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
int main(int argc, char** argv)
{
    srand((unsigned)time(NULL));
    int value = rand();
    printf("value is %d \n", value);
    return 0;
}