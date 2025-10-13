#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

long long int total_hits = 0;
pthread_mutex_t mutex;

void *calculate_pi(void *arg) {
    long long int local_tosses = *(long long*)arg;
    long long int local_hits = 0;
    // unsigned int seed = time(NULL) ^ pthread_self(); // xor
    unsigned int seed = (unsigned int)(time(NULL) ^ (uintptr_t)pthread_self());


    for(long long int i = 0; i < local_tosses; i++) {
      double x = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
      double y = (double)rand_r(&seed) / RAND_MAX * 2.0 - 1.0;
      if(x * x + y * y <= 1.0) {
        local_hits++;
      }
    }

    pthread_mutex_lock(&mutex);
    total_hits += local_hits;
    pthread_mutex_unlock(&mutex);
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

    // init mutex
    pthread_mutex_init(&mutex, NULL);
    
    // ===================== compute ===============================
    // long long int every_tosses = tosses / num_pthreads;
    // there maybe not divisible 
    long long int base = tosses / num_pthreads;
    long long int remain = tosses % num_pthreads;
    long long* arguments = (long long*)malloc(sizeof(long long) * num_pthreads);

    for(int i = 0; i < num_pthreads; i++) {
        arguments[i] = base + (i < remain ? 1 : 0);
        // pthread_create(&pthreads[i], attribute, function, argument);
        pthread_create(&pthreads[i], NULL, calculate_pi, &arguments[i]);
    }

    // wait
    for(int i = 0; i < num_pthreads; i++) {
        pthread_join(pthreads[i], NULL);
    }

    // ===================== end =================================
    pthread_mutex_destroy(&mutex);

    double pi = 4.0 * (double)total_hits / (double)tosses;
    printf("%lf\n", pi);

    free(pthreads);
    free(arguments);
    return 0;
}