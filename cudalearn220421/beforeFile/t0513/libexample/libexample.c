#include <stdio.h>
#include <unistd.h>
#include <dlfcn.h>
#include <string.h>

//puts function declaration
    //to intercept the original puts
    //we define a function with the exact same name and function signature as the original libc "puts" function
int puts(const char *message){
    //declare the function pointer new_puts that will point to the originally intended puts function.
    //As before with the intercepting function declaration this pointer's function signature must match the function signature of puts.
    int (*new_puts) (const char *message);
    int result;
    //void *dlsym(void *handle, const char *symbol);
    //dlsym通过句柄（void *handle）和连接符名称（const char *symbol）获取函数名或者变量名
    //获取一个函数
    //initializes our function pointer using the dlsym() function.
    //The RTLD_NEXT enum tells the dynamic loader API that we want to return the 
    //next instance of the function associated with the second argument(in this case puts) in the load order.
     
    new_puts = dlsym(RTLD_NEXT, "puts");
    //strcmp函数语法为“int strcmp(char *str1,char *str2)”，
    //其作用是比较字符串str1和str2是否相同，如果相同则返回0，
    //如果不同，前者大于后者则返回1，否则返回-1
    if(strcmp(message, "Hello world!n") == 0){
        result= new_puts("Goodbye, cruel world!n");
    }  else {
        result = new_puts(message);
    }
}