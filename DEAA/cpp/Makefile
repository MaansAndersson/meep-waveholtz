# Build the C++ cavity probe against this repo's own MEEP install.
#
# libmeep was built with mpic++ and has one unresolved external symbol
# (_MPI_Init), so mpic++ is required -- plain g++/c++ will not link.  Everything
# else (harminv, gsl, hdf5, mpb, lapack/blas, fftw) is already a recorded dylib
# dependency of libmeep or statically absorbed into it, so -lmeep is the whole
# link line.
#
# Paths are given explicitly rather than via pkg-config on purpose: this repo's
# setup.sh exports CPATH and LIBRARY_PATH pointing at build_*/…, which makes a
# bare `c++ -lmeep` appear to work and makes `pkg-config --cflags meep` return
# nothing (pkgconf drops -I dirs that are already in CPATH).  Being explicit
# keeps the build correct in a shell that has not sourced setup.sh.
#
#   make          build
#   make run      build and run serially
#   make mpi      build and run on 4 ranks (must print identical arrays)
#   make clean

MEEP_ROOT ?= $(abspath ../..)
PREFIX    := $(MEEP_ROOT)/build_meep

# `CXX ?=` would not take effect: make defines CXX itself, so ?= never fires.
ifeq ($(origin CXX),default)
  CXX := mpic++
endif
CXXFLAGS ?= -O3 -fstrict-aliasing -Wall -W -std=c++17

#CXXFLAGS := -g -O0 -fno-omit-frame-pointer -std=c++17 -Wall -Wextra \
#                  -fsanitize=address,undefined -fno-sanitize-recover=all

INCLUDES := -I$(PREFIX)/include
LDLIBS   := -L$(PREFIX)/lib -lmeep

# So the dylibs resolve without DYLD_LIBRARY_PATH.
COMMA  := ,
RPATHS := $(patsubst %,-Wl$(COMMA)-rpath$(COMMA)%, \
            $(PREFIX)/lib $(MEEP_ROOT)/build_gsl/lib $(MEEP_ROOT)/build_fftw/lib \
            $(MEEP_ROOT)/build_mpb/lib $(MEEP_ROOT)/build_libctl/lib)

TARGET := test_cavity
TARGET2 := pi
OUTDIR ?= ../notes/cavity_c
NT     ?= 1

.PHONY: all run mpi clean
all: $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDLIBS) $(RPATHS)

pi: $(TARGET2) 
$(TARGET2): $(TARGET2).cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LDLIBS) $(RPATHS)

run: $(TARGET)
	./$(TARGET) $(NT) $(OUTDIR)

mpi: $(TARGET)
	mpirun -n 4 ./$(TARGET) $(NT) $(OUTDIR)

clean:
	rm -f $(TARGET)
	rm -rf $(TARGET).dSYM
