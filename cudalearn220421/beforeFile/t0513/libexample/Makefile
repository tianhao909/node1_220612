
src=helloworld.c
bin=helloworld
so = libexample.so
all: $(bin) libexample.so libhook.so

$(bin) : $(src)
	gcc $(src) -o $(bin) 

libexample.so : libexample.c
	gcc libexample.c -o libexample.so -fPIC -shared -ldl -D_GNU_SOURCE

libhook.so : hook.c 
	gcc hook.c -o libhook.so -fPIC -shared -lssl -D_GNU_SOURCE



clean:
	rm $(bin) libexample.so libhook.so