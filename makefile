rwildcard=$(wildcard $1$2) $(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2))
src := $(call rwildcard,./src/,*.cpp)

# 產生可執行檔名稱，這裡假設主要入口是 main.cpp，所以產生 main
EXE = main
obj = $(patsubst %.cpp, %.o, $(src))

LDFLAGS = -lsfml-graphics -lsfml-window -lsfml-system -lsfml-network

INTELMAC_INCLUDE=-I/usr/local/include                         # Intel mac
APPLESILICON_INCLUDE=-I/opt/homebrew/opt/sfml@2/include       # Apple Silicon
UBUNTU_APPLESILICON_INCLUDE=-I/usr/include                    # Apple Silicon Ubuntu VM
UBUNTU_INTEL_INCLUDE=-I/usr/include                           # Intel Ubuntu VM

INTELMAC_LIB=-L/usr/local/lib                                 # Intel mac
APPLESILICON_LIB=-L/opt/homebrew/opt/sfml@2/lib               # Apple Silicon
UBUNTU_APPLESILICON_LIB=-L/usr/lib/aarch64-linux-gnu          # Apple Silicon Ubuntu VM
UBUNTU_INTEL_LIB=-L/usr/lib/x86_64-linux-gnu                  # Intel Ubuntu VM

# Detect platform
uname_s := $(shell uname -s)
uname_m := $(shell uname -m)

ifeq ($(uname_s),Darwin)
    ifeq ($(uname_m),arm64)
        PLATFORM_INCLUDE=$(APPLESILICON_INCLUDE)
        PLATFORM_LIB=$(APPLESILICON_LIB)
        PLATFORM_COMPILER=/usr/bin/clang++
    else
        PLATFORM_INCLUDE=$(INTELMAC_INCLUDE)
        PLATFORM_LIB=$(INTELMAC_LIB)
        PLATFORM_COMPILER=/usr/bin/clang++
    endif
else ifeq ($(uname_s),Linux)
    ifeq ($(uname_m),aarch64)
        PLATFORM_INCLUDE=$(UBUNTU_APPLESILICON_INCLUDE)
        PLATFORM_LIB=$(UBUNTU_APPLESILICON_LIB)
        PLATFORM_COMPILER=/usr/bin/g++
    else
        PLATFORM_INCLUDE=$(UBUNTU_INTEL_INCLUDE)
        PLATFORM_LIB=$(UBUNTU_INTEL_LIB)
        PLATFORM_COMPILER=/usr/bin/g++
    endif
endif

all: $(EXE)

$(EXE): $(src)
	$(PLATFORM_COMPILER) -std=c++17 $(PLATFORM_INCLUDE) -o $(EXE) $(src) $(LDFLAGS) $(PLATFORM_LIB)

.PHONY: clean
clean:
	rm -f $(EXE) $(obj)
