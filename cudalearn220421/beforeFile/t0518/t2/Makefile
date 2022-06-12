all: main libhook.so
main : main.cpp
	g++ main.cpp -o main 

libhook.so : hook.cpp
	g++ hook.cpp -shared -fpic -o libhook.so

#export LD_PRELOAD=./libpreload.so

clean:
	rm main libhook.so