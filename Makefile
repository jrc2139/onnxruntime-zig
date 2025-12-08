# ONNX Runtime Static Build Makefile
# Tested on: macOS ARM64 (M4), Linux x86_64

ONNXRUNTIME_VERSION := 1.23.2
ONNXRUNTIME_REPO := https://github.com/microsoft/onnxruntime.git
BUILD_DIR := onnxruntime-build
INSTALL_DIR := $(CURDIR)/deps/onnxruntime-static

# Detect OS and architecture
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    OS := macos
    BUILD_SUBDIR := MacOS
    ifeq ($(UNAME_M),arm64)
        ARCH := arm64
        CMAKE_ARCH := CMAKE_OSX_ARCHITECTURES=arm64
    else
        ARCH := x86_64
        CMAKE_ARCH := CMAKE_OSX_ARCHITECTURES=x86_64
    endif
else ifeq ($(UNAME_S),Linux)
    OS := linux
    BUILD_SUBDIR := Linux
    ARCH := $(UNAME_M)
    CMAKE_ARCH := CMAKE_POSITION_INDEPENDENT_CODE=ON
endif

# Build configuration
CONFIG := Release
PARALLEL_JOBS := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Paths
ORT_BUILD_PATH := $(BUILD_DIR)/build/$(BUILD_SUBDIR)/$(CONFIG)

.PHONY: all clone build install clean distclean help

all: install

help:
	@echo "ONNX Runtime Static Build"
	@echo ""
	@echo "Targets:"
	@echo "  clone    - Clone ONNX Runtime repository"
	@echo "  build    - Build static libraries"
	@echo "  install  - Install libraries to deps/onnxruntime-static"
	@echo "  clean    - Clean build artifacts"
	@echo "  distclean - Remove everything including cloned repo"
	@echo ""
	@echo "Detected: $(OS) $(ARCH)"
	@echo "ONNX Runtime version: $(ONNXRUNTIME_VERSION)"

clone: $(BUILD_DIR)/.cloned

$(BUILD_DIR)/.cloned:
	@echo "Cloning ONNX Runtime v$(ONNXRUNTIME_VERSION)..."
	git clone --depth 1 --branch v$(ONNXRUNTIME_VERSION) $(ONNXRUNTIME_REPO) $(BUILD_DIR)
	@touch $@

build: $(BUILD_DIR)/.built

$(BUILD_DIR)/.built: $(BUILD_DIR)/.cloned
	@echo "Building ONNX Runtime static libraries for $(OS) $(ARCH)..."
	@echo "This may take 5-10 minutes..."
	cd $(BUILD_DIR) && python3 tools/ci_build/build.py \
		--build_dir ./build/$(BUILD_SUBDIR) \
		--config $(CONFIG) \
		--parallel $(PARALLEL_JOBS) \
		--skip_tests \
		--compile_no_warning_as_error \
		--cmake_extra_defines \
			$(CMAKE_ARCH) \
			onnxruntime_BUILD_UNIT_TESTS=OFF \
			CMAKE_DISABLE_FIND_PACKAGE_Protobuf=ON \
			onnxruntime_USE_FULL_PROTOBUF=ON \
			onnxruntime_BUILD_SHARED_LIB=OFF
	@touch $@

install: $(INSTALL_DIR)/.installed

$(INSTALL_DIR)/.installed: $(BUILD_DIR)/.built
	@echo "Installing static libraries to $(INSTALL_DIR)..."
	@mkdir -p $(INSTALL_DIR)/lib
	@mkdir -p $(INSTALL_DIR)/include

	# Copy core ONNX Runtime static libraries
	cp $(ORT_BUILD_PATH)/libonnxruntime_*.a $(INSTALL_DIR)/lib/

	# Copy dependency libraries
	cp $(ORT_BUILD_PATH)/_deps/onnx-build/libonnx*.a $(INSTALL_DIR)/lib/
	cp $(ORT_BUILD_PATH)/_deps/protobuf-build/libprotobuf.a $(INSTALL_DIR)/lib/
	cp $(ORT_BUILD_PATH)/_deps/flatbuffers-build/libflatbuffers.a $(INSTALL_DIR)/lib/
	cp $(ORT_BUILD_PATH)/_deps/pytorch_cpuinfo-build/libcpuinfo.a $(INSTALL_DIR)/lib/
	-cp $(ORT_BUILD_PATH)/_deps/kleidiai-build/libkleidiai.a $(INSTALL_DIR)/lib/ 2>/dev/null || true
	# Copy RE2 library (required for Tokenizer/RegexFullMatch ops)
	cp $(ORT_BUILD_PATH)/_deps/re2-build/libre2.a $(INSTALL_DIR)/lib/

	# Copy abseil libraries
	@mkdir -p $(INSTALL_DIR)/lib/abseil
	find $(ORT_BUILD_PATH)/_deps/abseil_cpp-build -name "*.a" -exec cp {} $(INSTALL_DIR)/lib/abseil/ \;

	# Copy headers (match dynamic library structure: include/onnxruntime_c_api.h)
	cp -r $(BUILD_DIR)/include/onnxruntime $(INSTALL_DIR)/include/
	# Also copy C API headers to root for compatibility
	cp $(BUILD_DIR)/include/onnxruntime/core/session/onnxruntime_c_api.h $(INSTALL_DIR)/include/
	cp $(BUILD_DIR)/include/onnxruntime/core/session/onnxruntime_cxx_api.h $(INSTALL_DIR)/include/
	cp $(BUILD_DIR)/include/onnxruntime/core/session/onnxruntime_cxx_inline.h $(INSTALL_DIR)/include/
	cp $(BUILD_DIR)/include/onnxruntime/core/session/onnxruntime_float16.h $(INSTALL_DIR)/include/
	cp $(BUILD_DIR)/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h $(INSTALL_DIR)/include/
	cp $(BUILD_DIR)/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h $(INSTALL_DIR)/include/
	-cp $(BUILD_DIR)/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h $(INSTALL_DIR)/include/ 2>/dev/null || true

	# Create library list for linking
	@echo "# ONNX Runtime Static Libraries" > $(INSTALL_DIR)/libs.txt
	@echo "# Link order matters!" >> $(INSTALL_DIR)/libs.txt
	@ls -1 $(INSTALL_DIR)/lib/*.a | xargs -n1 basename >> $(INSTALL_DIR)/libs.txt
	@echo "" >> $(INSTALL_DIR)/libs.txt
	@echo "# Abseil libraries" >> $(INSTALL_DIR)/libs.txt
	@ls -1 $(INSTALL_DIR)/lib/abseil/*.a | xargs -n1 basename >> $(INSTALL_DIR)/libs.txt

	@touch $@
	@echo ""
	@echo "Installation complete!"
	@echo "Libraries: $(INSTALL_DIR)/lib/"
	@echo "Headers:   $(INSTALL_DIR)/include/"
	@echo ""
	@du -sh $(INSTALL_DIR)/lib/

clean:
	rm -rf $(BUILD_DIR)/build
	rm -f $(BUILD_DIR)/.built

distclean:
	rm -rf $(BUILD_DIR)
	rm -rf $(INSTALL_DIR)

# Debug target to show detected configuration
info:
	@echo "OS:           $(OS)"
	@echo "Architecture: $(ARCH)"
	@echo "Build dir:    $(ORT_BUILD_PATH)"
	@echo "Install dir:  $(INSTALL_DIR)"
	@echo "CMake flags:  $(CMAKE_ARCH)"
	@echo "Parallel:     $(PARALLEL_JOBS) jobs"
