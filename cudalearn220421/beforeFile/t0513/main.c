#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

//动态链接库路径
#define LIB_CALCULATE_PATH "./libcalculate.so"

//函数指针
typedef  int (*CAC_FUNC)(int,int);

int main(){


    //void *dlsym(void *handle, const char *symbol);
    void *handle;
    char *error;
    CAC_FUNC cac_func = NULL;

    //void *dlopen(const char *filename, int flag); 
    //dlopen以指定模式打开指定的动态连接库文件（const char *filename），
    //并返回一个句柄（int flag）给调用进程
    //dlopen打开模式如下：  RTLD_LAZY 暂缓决定，等有需要时再解出符号 
    //　RTLD_NOW 立即决定，返回前解除所有未决定的符号。
    //打开动态链接库
    handle = dlopen(LIB_CALCULATE_PATH, RTLD_LAZY);  
    if(!handle){
        //char *dlerror(void); dlerror返回出现的错误
        fprintf(stderr, "%s\n", dlerror());
        exit(EXIT_FAILURE);
    }  

    //清除之前存在的错误
    dlerror();

    //void *dlsym(void *handle, const char *symbol);
    //dlsym通过句柄（void *handle）和连接符名称（const char *symbol）获取函数名或者变量名
    //获取一个函数
    *(void **) (&cac_func) = dlsym(handle, "add");
    if ((error = dlerror()) != NULL) {
        fprintf(stderr, "%s\n", error);
        exit(EXIT_FAILURE);
    }
    printf("add: %d\n", (*cac_func)(2,7));

    cac_func = (CAC_FUNC)dlsym(handle, "sub");
    printf("sub: %d\n", cac_func(9,2));

    //void *dlsym(void *handle, const char *symbol);
    //dlsym通过句柄（void *handle）和连接符名称（const char *symbol）获取函数名或者变量名
    //获取一个函数
    cac_func = (CAC_FUNC)dlsym(handle, "mul");
    printf("mul: %d\n", cac_func(3, 2));

    cac_func = (CAC_FUNC)dlsym(handle, "div");
    printf("div: %d\n", cac_func(8,2));

     //关闭动态链接库
     dlclose(handle);
     exit(EXIT_SUCCESS);

}
