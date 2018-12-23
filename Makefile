AMDGPU_TARGETS=-amdgpu-target=gfx803 -amdgpu-target=gfx900
CXXFLAGS=-O2 -std=c++17 -Wall
TARGETS=saxpy-hc saxpy-cl
HCC_PREFIX=/opt/rocm/hcc

all: $(TARGETS)
clean:
	rm -f $(TARGETS)

%-hc: %-hc.cpp *.hpp
	$(HCC_PREFIX)/bin/hcc -hc $(CXXFLAGS) $(AMDGPU_TARGETS) $< -o $@

%-cl: %-cl.cpp *.hpp
	$(CXX) $(CXXFLAGS) -lOpenCL $< -o $@
