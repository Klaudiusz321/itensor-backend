CC = gcc
CFLAGS = -Wall -Wextra -fPIC -O2 -g
LDFLAGS = -shared -lm

# Source files
SRCS = mhd.c \
       mhd_solver.c \
       mhd_conditions.c \
       mhd_magnetic.c \
       mhd_fluid.c \
       mhd_advance.c \
       mhd_utils.c \
       mhd_visualization.c

# Object files
OBJS = $(SRCS:.c=.o)

# Target library
TARGET = libmhd.so

# Default target
all: $(TARGET)

# Create shared library
$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

# Compile source files into object files
%.o: %.c mhd.h
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGET) test_mhd test_mhd.c mhd_slice.csv

# Install the library (usually to /usr/local/lib)
install: $(TARGET)
	install -d $(DESTDIR)/usr/local/lib
	install -m 755 $(TARGET) $(DESTDIR)/usr/local/lib
	install -d $(DESTDIR)/usr/local/include
	install -m 644 mhd.h $(DESTDIR)/usr/local/include

# Test the library with a simple example
test: $(TARGET) test_mhd
	./test_mhd

# Compile the test program
test_mhd: test_mhd.c $(TARGET)
	$(CC) -o test_mhd test_mhd.c -L. -lmhd -Wl,-rpath,.

# Example test file (separate file, easy to modify)
test_mhd.c: test_mhd.c
	@echo "Creating test file..."
	echo '#include <stdio.h>' > test_mhd.c
	echo '#include "mhd.h"' >> test_mhd.c
	echo 'int main() {' >> test_mhd.c
	echo '    printf("Starting MHD Simulation Test...\\n");' >> test_mhd.c
	echo '    MHDSimulation *sim = mhd_initialize(50, 50, 50);' >> test_mhd.c
	echo '    if (!sim) {' >> test_mhd.c
	echo '        fprintf(stderr, "Failed to initialize simulation\\n");' >> test_mhd.c
	echo '        return 1;' >> test_mhd.c
	echo '    }' >> test_mhd.c
	echo '    mhd_initialize_magnetic_field(sim, 0.1, 0.0, 0.0);' >> test_mhd.c
	echo '    mhd_set_boundary_type(sim, PERIODIC);' >> test_mhd.c
	echo '    mhd_set_numerical_method(sim, RUNGE_KUTTA_4);' >> test_mhd.c
	echo '    mhd_set_total_time(sim, 1.0);' >> test_mhd.c
	echo '    mhd_set_time_step(sim, 0.01);' >> test_mhd.c
	echo '    printf("Running MHD simulation...\\n");' >> test_mhd.c
	echo '    mhd_run_simulation(sim);' >> test_mhd.c
	echo '    mhd_export_slice_data(sim, 2, 25, "mhd_slice.csv");' >> test_mhd.c
	echo '    mhd_free(sim);' >> test_mhd.c
	echo '    printf("Simulation completed!\\n");' >> test_mhd.c
	echo '    return 0;' >> test_mhd.c
	echo '}' >> test_mhd.c

.PHONY: all clean install test
