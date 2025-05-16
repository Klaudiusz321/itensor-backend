CC = gcc
CFLAGS = -O2 -Wall -Wextra -I.
LDFLAGS = -shared
LIBS = -lm

SRC = mhd.c \
      mhd_solver.c \
      mhd_utils.c \
      mhd_magnetic.c \
      mhd_conditions.c \
      mhd_visualization.c \
      mhd_fluid.c \
      mhd_advance.c \
      mhd_metrics.c \
      mhd_run_step.c

OBJ = $(SRC:.c=.o)

ifeq ($(OS),Windows_NT)
    TARGET = libmhd.dll
else
    TARGET = libmhd.so
endif

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

test: test.c $(TARGET)
	$(CC) $(CFLAGS) -o test test.c -L. -lmhd $(LIBS)

.PHONY: all clean test
