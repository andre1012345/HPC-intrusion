# Compilers
CXX = mpicxx
CXXFLAGS = -fopenmp -O3 -Wall -Iinclude

# Directories
SRC_DIR = src
BIN_DIR = bin

# Target executable name
TARGET = $(BIN_DIR)/ensemble_classifier

# Find all .cpp files in src
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
# Convert .cpp filenames to .o (object) files
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(BIN_DIR)/%.o)

# Default rule
all: $(TARGET)

# Link the object files to create the binary
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

# Compile .cpp files to .o files
$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up the build
clean:
	rm -rf $(BIN_DIR)/*

.PHONY: all clean