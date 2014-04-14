INCDIR = -I.
DBG    = -g
OPT    = -O3
CPP    = g++
CFLAGS = $(DBG) $(OPT) $(INCDIR)

.cpp.o:
	$(CPP) $(CFLAGS) -c $< -o $@

all: depth

depth: depth.cpp
	$(CPP) $(CFLAGS) -o depth depth.cpp

clean:
	/bin/rm -f depth *.o

clean-all: clean
	/bin/rm -f *~



