all: test libpreload.so
test : test.c
	gcc test.c -o test 

libpreload.so : preload.c
	gcc preload.c -shared -fpic -o libpreload.so

#export LD_PRELOAD=./libpreload.so

clean:
	rm test libpreload.so