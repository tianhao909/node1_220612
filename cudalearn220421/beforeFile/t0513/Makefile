o_target=libcalculate.so
src=calculate.c 
bin=main

all: $(bin) $(o_target)

$(bin) : main.c
	gcc -o $(bin) main.c -ldl
#	gcc -o $(bin) main.c -ldl
#	gcc -rdynamic -o $(bin) main.c -ldl


$(o_target) : $(src)
	gcc -fPIC -shared $(src) -o $(o_target)
#	gcc -fPIC -shared calculate.c -o libcalculate.so





clean:
	rm libcalculate.so main
