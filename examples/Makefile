CXX := g++
CXXFLAGS := -Wall -Werror -std=c++20

BIN_DIR := ./bin
SRC_DIR := ./
INC_DIR := ../

SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
HEADERS := $(wildcard $(INC_DIR)/*.h)
OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BIN_DIR)/%.o,$(SOURCES))
TARGET := program

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDLIBS)

$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	$(CXX) -c $< $(CXXFLAGS) -I $(INC_DIR) -o $@ 

clean:
	rm -rf $(OBJECTS) && rm -f $(TARGET)

.PHONY: clean