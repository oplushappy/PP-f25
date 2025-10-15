CC = gcc
CFLAGS = -O3 -march=native -mtune=native -mavx2 -mfma -ffast-math -Wall
LDFLAGS = -pthread

pi.out: pi.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f pi.out
