#include <mclq/random.h>
#include <stdint.h>
#include <sys/random.h>

static uint64_t _buffer[N_RANDOM_BYTES / sizeof(uint64_t)];
static size_t _idx = N_RANDOM_BYTES / sizeof(uint64_t);

static void replenish(void) {
	int x = getrandom(_buffer, sizeof(_buffer), 0);
	(void)x;
	_idx = 0;
}

static inline double _random_unit_interval() {
	if (_idx >= (N_RANDOM_BYTES / sizeof(uint64_t))) {
		replenish();
	}

	// Take 53 bits for double precision (IEEE 754)
	// This avoids the bias inherent in dividing a 32-bit int
	uint64_t r = _buffer[_idx++] & 0x1FFFFFFFFFFFFFLLU;
	return (double)r / (double)(1LLU << 53);
}

float randfloat(float lb, float ub) {
	return lb + (float)(_random_unit_interval() * (double)(ub - lb));
}

double randdouble(double lb, double ub) {
	return lb + (_random_unit_interval() * (ub - lb));
}

int randint(int lb, int ub) {
	return lb + (int)(_random_unit_interval() * (double)(ub - lb));
}
