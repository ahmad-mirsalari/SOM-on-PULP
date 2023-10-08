#ifndef _CONFIG_MATMUL_
#define _CONFIG_MATMUL_
#include "param.h"

#ifdef FABRIC
#define DATA_LOCATION
#else
#define DATA_LOCATION __attribute__((section(".data_l1")))
#endif

#if M_g == 8
   #define No_for
#endif
#if N_g == w_b
 #define once_load
#endif

#ifndef IN_ORDER

    #if R_g > NUM_CORES
        #error "The number of cores must be greater than the number of SOMs...!!!" 
    #endif
#endif
#if R_g ==1 
    #ifndef IN_ORDER
     #error "Enalbe IN_ORDER flag...!!!"
    #endif
#endif 
// #ifdef FABRIC
// #define DATA_LOCATION
// #else
// #define DATA_LOCATION __attribute__((section(".data_l1")))
// #endif

#ifdef FP32
typedef float Pixel;
#elif defined(FP16)
typedef float16 Pixel; //__attribute__ ((aligned (2)));
typedef float16 VPixel    __attribute__((vector_size (4)));
#undef USE_INTRINSICS
#elif defined(FP16ALT)
typedef float16alt Pixel;
typedef float16alt VPixel    __attribute__((vector_size (4)));
#undef USE_INTRINSICS
#elif defined(FP8)
typedef float8 Pixel;
typedef float8 VPixel    __attribute__((vector_size (4)));
#undef USE_INTRINSICS
#endif


// #define IN_ORDER


void SOM_Train(Pixel * __restrict__ SOM_N, Pixel * __restrict__ Inp, int N, int M, int I, int Epoch);
Pixel SOM_Test(Pixel * __restrict__ SOM_N, Pixel * __restrict__ Inp, int N, int M, int K);

#endif
