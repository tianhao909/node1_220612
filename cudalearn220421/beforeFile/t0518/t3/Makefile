all: malloc  prehook.so
malloc : malloc.c
	gcc malloc.c -o malloc

prehook.so : premalloc.c
	gcc premalloc.c -o prehook.so -fPIC -shared -ldl 

# libhook.so : hook.cpp
# 	g++ hook.cpp -shared -fpic -o libhook.so

#export LD_PRELOAD=./libpreload.so
#export LD_PRELOAD=./libhook.so

clean:
	rm malloc prehook.so