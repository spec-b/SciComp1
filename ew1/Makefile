CC=gcc
CFLAGS=`pkg-config --cflags gtk+-3.0` -O2
LIBS=`pkg-config --libs gtk+-3.0` -lm
MPICC=mpicc

BIN_DIR=bin

GUI_DIR=gui_c

COORD=coordinator.c 
WORKER=c_workers.c

PYDIR=python_ml
PY_MAIN=$(PYDIR)/ml_pipeline.py
PY_FILES=$(PYDIR)/data_fetch.py $(PYDIR)/bayes_model.py

# Create virtual environment and install dependencies
install-deps:
	python3 -m venv venv
	source venv/bin/activate && pip install -r requirements.txt

all: install-deps $(BIN_DIR)/coordinator $(BIN_DIR)/worker gui


GUI_MAIN=$(GUI_DIR)/main.py

all: $(BIN_DIR)/coordinator $(BIN_DIR)/worker gui

$(BIN_DIR)/coordinator: $(COORD)
	$(MPICC) -o $@ $^

$(BIN_DIR)/worker: $(WORKER)
	$(MPICC) -o $@ $^

gui:
	@echo "Running Python GUI..."
	source venv/bin/activate && python3 $(GUI_MAIN)

run-gui:
	source venv/bin/activate && python3 $(GUI_MAIN)

clean:
	rm -f $(BIN_DIR)/* $(PYDIR)/*.npy $(PYDIR)/forecast.png




