#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>


typedef struct {
    long long int local_tosses;
    long long int local_hit;
} Arg;

uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}


void *calculate_pi(void *arg) {
    Arg* data = (Arg*) arg;
    long long int local_tosses = data->local_tosses;
    long long int local_hits = 0;
    // unsigned int seed = time(NULL) ^ pthread_self(); // xor
    uint32_t state = (uint32_t)(time(NULL) ^ (uintptr_t)pthread_self());

    const float inverse_uint32_max = 1.0f / 4294967296.0f; // 2^32

    for(long long int i = 0; i < local_tosses; i++) {
      float x = (float)xorshift32(&state) * inverse_uint32_max * 2.0f - 1.0f;
      float y = (float)xorshift32(&state) * inverse_uint32_max * 2.0f - 1.0f;
      if(x * x + y * y <= 1.0f) {
        local_hits++;
      }
    }

    data->local_hit = local_hits;
    return NULL;
}

int main(int argc, char* argv[]) {
    if(argc != 3) {
        fprintf(stderr, "Usage: %s <threads:int> <tosses:long long>\n", argv[0]);
        return 1;
    }
    // ===================== init =================================
    int num_pthreads = atoi(argv[1]);
    long long int tosses = atoll(argv[2]);

    // pthread_t pthreads[num_pthreads];
    pthread_t * pthreads = (pthread_t*)malloc(sizeof(pthread_t) * num_pthreads);

    
    // ===================== compute ===============================
    // long long int every_tosses = tosses / num_pthreads;
    // there maybe not divisible 
    long long int base = tosses / num_pthreads;
    long long int remain = tosses % num_pthreads;
    Arg* args = malloc(sizeof(Arg) * num_pthreads);

    for(int i = 0; i < num_pthreads; i++) {
        args[i].local_tosses = base + (i < remain ? 1 : 0);
        args[i].local_hit = 0;
        // pthread_create(&pthreads[i], attribute, function, argument);
        pthread_create(&pthreads[i], NULL, calculate_pi, &args[i]);
    }

    // wait
    for(int i = 0; i < num_pthreads; i++) {
        pthread_join(pthreads[i], NULL);
    }

    long long int total_hits = 0;
    for(int i = 0; i < num_pthreads; i++) {
        total_hits += args[i].local_hit;
    }

    // ===================== end =================================
    double pi = 4.0 * (double)total_hits / (double)tosses;
    printf("%lf\n", pi);

    free(pthreads);
    free(args);
    return 0;
}