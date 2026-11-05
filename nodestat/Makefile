CC ?= cc
CFLAGS ?= -O2 -g -Wall -Wextra -Wno-unused-parameter
override CFLAGS += -std=c23 -pthread

all: nodestat mock_nvml.so

nodestat: nodestat.c index.html
	$(CC) $(CFLAGS) -o $@ nodestat.c -ldl -lm

mock_nvml.so: mock_nvml.c
	$(CC) $(CFLAGS) -shared -fPIC -o $@ mock_nvml.c -lm

clean:
	rm -f nodestat mock_nvml.so

.PHONY: all clean
