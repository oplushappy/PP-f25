#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>
#include <stdalign.h> 


typedef struct {
    long long int local_tosses;
    long long int local_hit;
} Arg;

__m256i xorshift32_vec(__m256i state) {
    __m256i x = state;
    x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 13));
    x = _mm256_xor_si256(x, _mm256_srli_epi32(x, 17));
    x = _mm256_xor_si256(x, _mm256_slli_epi32(x, 5));
    return x;
}

void *calculate_pi(void *arg) {
    Arg* data = (Arg*) arg;
    long long int local_tosses = data->local_tosses;
    long long int local_hits = 0;
    
    uint32_t base = (uint32_t)(time(NULL) ^ (uintptr_t)pthread_self());
    alignas(32) uint32_t s[8];
    for(int i = 0; i < 8; i++) {
        s[i] = base + i * 767796405u;
    }
    __m256i state = _mm256_load_si256((const __m256i*)s); // __m256i state = _mm256_setr_epi32(s0, s1, s2, s3, s4, s5, s6, s7);
    

    // const float inverse_uint32_max = 1.0f / 4294967296.0f; // 2^32
    // const __m256 v_scale = _mm256_set1_ps(inverse_uint32_max * 2.0f);
    // const __m256 v_bias = _mm256_set1_ps(-1.0f);
    const __m256 v_scale = _mm256_set1_ps(1.0f / 2147483648.0f); // 2^31
    const __m256 v_one = _mm256_set1_ps(1.0f);

    long long int i = 0;
    for(; i + 7 < local_tosses; i+=8) {
      state = xorshift32_vec(state);
      // pitfall, _mm256_cvtepi32_ps: use int to float , not unsigned int to float 
      // __m256 x = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(state), v_scale), v_bias);
      __m256 x = _mm256_mul_ps(_mm256_cvtepi32_ps(state), v_scale);
       
      state = xorshift32_vec(state);
      // __m256 y = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(state), v_scale), v_bias);
      __m256 y = _mm256_mul_ps(_mm256_cvtepi32_ps(state), v_scale);

      __m256 distance = _mm256_fmadd_ps(x, x, _mm256_mul_ps(y, y)); 

      __m256 cmp = _mm256_cmp_ps(distance, v_one, _CMP_LE_OQ);
      int mask =_mm256_movemask_ps(cmp);
      local_hits += __builtin_popcount(mask);
    }

    if (i < local_tosses) {
        uint32_t tail = base ^ 0x9E3779B9u;
        const float scale = 1.0f / 4294967296.0f * 2.0f;
        const float bias = -1.0f;
        for (; i < local_tosses; i++) {
            // x
            tail ^= tail << 13; tail ^= tail >> 17; tail ^= tail << 5;
            float xf = (float)tail * scale + bias;
            // y
            tail ^= tail << 13; tail ^= tail >> 17; tail ^= tail << 5;
            float yf = (float)tail * scale + bias;

            local_hits += (xf*xf + yf*yf <= 1.0f);
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