CPP    = g++
WINDRES= windres
RM     = rm -f
OBJS   = src/main.o \
         src/game.o \
         src/player.o \
         AppResource.res

LIBS   = -static -lglut32 -lglu32 -lopengl32 -lwinmm -lgdi32
CFLAGS = -DGLUT_STATIC

.PHONY: Projeto.exe clean clean-after

all: Projeto.exe

clean:
	$(RM) $(OBJS) Projeto.exe

clean-after:
	$(RM) $(OBJS)

Projeto.exe: $(OBJS)
	$(CPP) -Wall -s -o $@ $(OBJS) $(LIBS)

src/main.o: src/main.cpp src/game.h src/player.h
	$(CPP) -Wall -s -c $< -o $@ $(CFLAGS)

src/game.o: src/game.cpp
	$(CPP) -Wall -s -c $< -o $@ $(CFLAGS)

src/player.o: src/player.cpp src/player.h
	$(CPP) -Wall -s -c $< -o $@ $(CFLAGS)

AppResource.res: AppResource.rc
	$(WINDRES) -i AppResource.rc -J rc -o AppResource.res -O coff

