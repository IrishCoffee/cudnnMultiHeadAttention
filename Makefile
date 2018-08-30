CUDA_PATH ?= "/usr/local/cuda-9.0"

GPP ?= g++
NVCC        := $(CUDA_PATH)/bin/nvcc 

NVCCFLAGS   := -default-stream per-thread -std=c++11 -m64 -Xcompiler -fopenmp -O2
CCFLAGS     :=
LDFLAGS     :=

# Debug build flags
ifeq ($(dbg),1)
      NVCCFLAGS += -g -G
      BUILD_TYPE := debug
else
      BUILD_TYPE := release
endif

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(EXTRA_NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(EXTRA_CCFLAGS))

SAMPLE_ENABLED := 1

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))
ALL_LDFLAGS += $(addprefix -Xlinker ,$(EXTRA_LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I$(CUDA_PATH)/samples/common/inc -I$(CUDA_PATH)/include -I .  -I/matrix -I./kernel
LIBRARIES := -L$(CUDA_PATH)/lib64 -lcublas -lcudnn

################################################################################
# Gencode arguments
SM ?= 61
GENCODE_FLAGS += -gencode arch=compute_$(SM),code=compute_$(SM) 
OBJ_DIR = ./obj
################################################################################

# Target rules
all: build

build: main

OBJS := $(OBJ_DIR)/main.o 

main: $(OBJS) 
	$(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

$(OBJ_DIR)/main.o: main.cu
	$(NVCC) $(INCLUDES) $(LIBRARIES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

clean:
	rm main $(OBJ_DIR)/*.o

