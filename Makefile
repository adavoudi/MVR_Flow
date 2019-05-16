CC = g++
CFLAGS = -g -Wall
SRCS = main.cc
PROG = main

OPENCV = `pkg-config opencv4 --cflags --libs`
LIBAV = `pkg-config --cflags --libs libavformat libswscale libavutil libavcodec`
# LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) -o $(PROG) $(SRCS) -L $(OPENCV) -L $(LIBAV)
