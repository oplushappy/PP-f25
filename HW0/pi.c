#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// pi = 4 x N /T;
int main() {
    long long int num_in_square = 100000000;  // 1e8
    long long int num_in_circle = 0;

    srand(time(NULL));

    for (long long int i = 0; i < num_in_square; i++) {
        double x = (double)rand() / RAND_MAX * 2.0 - 1.0;
        double y = (double)rand() / RAND_MAX * 2.0 - 1.0;
        if (x * x + y * y <= 1.0)
            num_in_circle++;
    }

    double pi = 4.0 * num_in_circle / (double)num_in_square;
    printf("%lf\n", pi);
    return 0;
}
