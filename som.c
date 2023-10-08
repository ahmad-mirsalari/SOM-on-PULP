#include "pmsis.h"
#include "config.h"

#include <math.h>
#include <stdbool.h>
#define ABS(x) (((x) < 0)? (-(x)) : (x))

#ifdef FP32
Pixel FABS(Pixel x){
int temp = (*((int *)(&x)) & 0X7FFFFFFF);
return *((Pixel *)(&temp));
}
#else
  #ifdef VECTORIAL
  
    #ifdef FP8
    VPixel FABS(VPixel x){
      int temp = (*((int *)(&x)) & 0X7F7F7F7F);
      return *((VPixel *)(&temp));
    }
    #else
    VPixel FABS(VPixel x){
      int temp = (*((int *)(&x)) & 0X7FFF7FFF);
      return *((VPixel *)(&temp));
    }
    #endif
  #else
    #ifdef FP8
      Pixel FABS(Pixel x){
        char temp = (*((char *)(&x)) & 0X7F);
        return *((Pixel *)(&temp));
        }
    #else
      Pixel FABS(Pixel x){
        short temp = (*((short *)(&x)) & 0X7FFF);
        return *((Pixel *)(&temp));
        }
    #endif
  #endif

#endif




#ifdef FP32
  #define MANTISSA_BITS 23
  #define EXPONENT_BITS 8
  #define EXPONENT_BIAS 127
  #define INT_TYPE int
#endif

#ifdef FP16
  #define MANTISSA_BITS 10
  #define EXPONENT_BITS 5
  #define EXPONENT_BIAS 15
  #define INT_TYPE short
#endif

#ifdef FP16ALT
  #define MANTISSA_BITS 7
  #define EXPONENT_BITS 8
  #define EXPONENT_BIAS 127
  #define INT_TYPE short
#endif

#ifdef FP8
  #define MANTISSA_BITS 2
  #define EXPONENT_BITS 5
  #define EXPONENT_BIAS 15
  #define INT_TYPE char
#endif
// Compute the value 2^(-x)
Pixel neg_power_of_two(unsigned INT_TYPE x) {
  INT_TYPE val = EXPONENT_BIAS - x;
  unsigned INT_TYPE exp = *((unsigned INT_TYPE *) &val);
  unsigned INT_TYPE res = exp << MANTISSA_BITS;
  return *((Pixel *) &res);
}

#ifndef FABRIC
DATA_LOCATION Pixel minimum[NUM_CORES];
DATA_LOCATION int myindex[NUM_CORES];

DATA_LOCATION Pixel min_start =448;
#if defined (once_load) 
  #ifndef IN_ORDER
    DATA_LOCATION Pixel SOM[R_g * w_b * M_g];
    
    DATA_LOCATION Pixel INP[2*R_g*inp_b*M_g];
  #else
    DATA_LOCATION Pixel SOM[ w_b * M_g];//2* w_b * M
    
    DATA_LOCATION Pixel INP[2*inp_b*M_g];//2*inp_b*M
  #endif
#elif defined(IN_ORDER)
  DATA_LOCATION Pixel SOM[2 *w_b * M_g];//2* w_b * M
 
  DATA_LOCATION Pixel INP[2 *inp_b*M_g];//2*inp_b*M
#else
  DATA_LOCATION Pixel SOM[2 *R_g *w_b * M_g];//2* w_b * M
  
  DATA_LOCATION Pixel INP[2* R_g *inp_b*M_g];//2*inp_b*M
#endif

int find_BMU_index(Pixel * __restrict__ x, Pixel * __restrict__ SOM_N, int N, int M, int s){

  
  pi_cl_dma_cmd_t commandA;
  pi_cl_dma_cmd_t commandB;

  #ifdef VECTORIAL
  VPixel temp;
  VPixel Inpv0, Inpv1, Sv0, Sv1;
  Pixel sum =0; 
  #else
  Pixel temp = 0;
  #endif
  int location =0;
  int start_par =0;
  int end =0;
  int ava_cores =0;
  int x_offset = 0;
  Pixel minu = min_start;
  #ifdef once_load
    int double_buffer =1;
  #else
    int double_buffer =2;
  #endif
  int core_id = pi_core_id();
  minimum[core_id] =min_start;
  // if (core_id ==0)
  //   for (int j = 0; j < M; j++) 
  //   {
  //     printf("core id %d x %d is %f \n",core_id,j, x[j]);
  //   }

 
  // if (s ==1)
  // printf("snm %d  \n",j, (s * N * M));


  // Load the first parts of SOMs
  #ifndef once_load
    if (core_id == 0) // load the first som
      {
        pi_cl_dma_cmd_2d((int) (SOM_N ) , (int) SOM ,w_b*M*sizeof(SOM_N[0]),
        w_b*M*sizeof(SOM_N[0]),w_b*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_EXT2LOC, &commandB);
        pi_cl_dma_cmd_wait(&commandB);
        
      }
    #ifndef IN_ORDER // load the second som if we run both SOMs in parallel
      if (core_id == 0)
        {
          pi_cl_dma_cmd_2d((int) (SOM_N + ( N * M) ), (int) (SOM + double_buffer * w_b * M ) ,w_b*M*sizeof(SOM_N[0]),
          w_b*M*sizeof(SOM_N[0]),w_b*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_EXT2LOC, &commandA);
          pi_cl_dma_cmd_wait(&commandA);
        }
      #endif
  #endif


  #ifdef IN_ORDER
    ava_cores = NUM_CORES;
  #else
    ava_cores = NUM_CORES/2;
  #endif

  // offset for the correct block of the L1 input for each cores
   x_offset =((core_id/(ava_cores)) * 2 * inp_b * M);


   #ifdef once_load
    int offset =0;
    int offset1 = w_b;
   #endif

  #ifndef once_load // we need to iterate on the whole SOM network and process wb data
   for (int counter =0; counter < N / w_b; counter++)
    {
  #endif


  #if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif 
  
  // load the next part of SOMs with DMA while processing the current data
  #ifndef once_load 

      if (core_id==0 && counter != (N/w_b)-1)
        {

          // printf("SOM data for next iter from %d of input l2 put   %d l1\n", (N * M) + ((counter +1) * w_b*M),(((counter +1) % 2)+ 2)* w_b*M);
      
          pi_cl_dma_cmd_2d((int) (SOM_N + ((counter +1) * w_b*M)) , (int) (SOM + ((counter +1) % 2)* w_b*M) ,w_b*M*sizeof(SOM[0]),
          w_b*M*sizeof(SOM[0]),w_b*M*sizeof(SOM[0]),PI_CL_DMA_DIR_EXT2LOC, &commandA);
        #ifndef IN_ORDER
            pi_cl_dma_cmd_2d((int) (SOM_N + (N * M) + ((counter +1) * w_b*M)) , (int) (SOM + (((counter +1) % 2)+ 2)* w_b*M) ,w_b*M*sizeof(SOM[0]),
            w_b*M*sizeof(SOM[0]),w_b*M*sizeof(SOM[0]),PI_CL_DMA_DIR_EXT2LOC, &commandB);
        #endif
        }
  #endif

  #ifndef once_load
    #ifndef IN_ORDER  
        int offset =(int)(counter / 2) * 2 * w_b;
        int offset1 = 2 * w_b;
    #else
        int offset =(int)(counter / 2) * 2 * w_b;
        int offset1 = 0;
    #endif
  // #else
  //   #ifndef IN_ORDER
  //     int offset =0;
  //     int offset1 = w_b;
  //   #else
  //     int offset =0;
  //     int offset1 = w_b;
  //   #endif
  #endif



  #ifndef once_load

    #ifdef IN_ORDER
      #if NUM_CORES > 1
        int blockSize = (w_b+ava_cores-1)/ava_cores;
        start_par = ((counter %2) * w_b) + core_id*blockSize ;
        end = start_par + blockSize < (2*w_b)? start_par + blockSize : (2*w_b);
      #else
         start_par = ((counter %2) * w_b);
         end = ((counter %2 ) +1) * w_b;
      #endif

    #else

      #if NUM_CORES > 1
        int blockSize = (w_b+ava_cores-1)/ava_cores;
        if (core_id < (ava_cores)){

          start_par = ((counter %2) * w_b) + core_id*blockSize ;
          end = start_par + blockSize < (2*w_b)? start_par + blockSize : (2*w_b);
        }
        else
        {
          start_par = ((counter %2) * w_b) + (core_id -(ava_cores)) *blockSize + (2*w_b) ;
          end = start_par + blockSize < (4*w_b)? start_par + blockSize : (4*w_b);
        }

      #else //  only one core
        start_par = ((counter %2) * w_b);
        end = ((counter %2 ) +1) * w_b;
      #endif

    #endif
  
  #else

    #ifndef IN_ORDER
      #if NUM_CORES > 1
          int blockSize = (w_b+(ava_cores)-1)/(ava_cores);
          if (core_id < (ava_cores)){

            start_par = core_id*blockSize ;
            end = start_par + blockSize < (w_b)? start_par + blockSize : (w_b);
          }
          else
          {
            start_par =  (core_id -(ava_cores)) *blockSize + (w_b) ;
            end = start_par + blockSize < (2*w_b)? start_par + blockSize : (2*w_b);
          }

        #else // we don't have this case, because we don't run both SOMs on one core with ndef in_order flag
          start_par = 0;
          end = 0;
        #endif
    #else // IN_ORDER 
      #if NUM_CORES > 1
        int blockSize = (w_b+ava_cores-1)/ava_cores;
        start_par =  core_id*blockSize ;
        end = start_par + blockSize < (w_b)? start_par + blockSize : (w_b);
      #else
         start_par = 0;
         end =  w_b;
      #endif
    #endif

  #endif

      // printf("core id is %d start %d end %d \n", core_id, start_par, end);

      // printf("index function core %d start %d end %d \n", core_id, start_par, end);
      //for (int i = ((counter %2) * w_b) + start_par; i < (((counter %2 ) +1) * w_b; i++) 
      for (int i =  start_par; i < end; i++)
      {

        #ifdef VECTORIAL
          #ifdef FP8
            temp = (VPixel) {0, 0, 0, 0};
            // for (int j = 0; j < M ; j+=8) 
            //   {
              int j =0;
                // if (core_id== 4)
                  // printf("core id is %d i %d j %d som %f x[%d] %f\n", core_id ,i, j, SOM[ i * M + j],((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j, x[((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j] );
                // printf("i   %d j is %d som %f and x is %f \n", i, j, SOM[i * M +j], x[x_offset + j]);
                Inpv0  = *((VPixel *) &x[x_offset + j]);
                Sv0  = *((VPixel *) &SOM[ i * M + j]);

                Inpv1  = *((VPixel *) &x[x_offset + j+4]);
                Sv1  = *((VPixel *) &SOM[ i * M + j+4]);

                temp += FABS(Sv0 - Inpv0);
                // printf("temp %f , %f , %f, %f \n",temp[0] , temp[1] , temp[2] , temp[3]);
                temp += FABS(Sv1 - Inpv1);
                // printf("temp %f , %f , %f, %f \n",temp[0] , temp[1] , temp[2] , temp[3]);

              //}
              sum = temp[0] + temp[1] + temp[2] + temp[3];
              // printf("sum %f \n", sum);
            // printf("temp is %f i is %d\n", sum, ( offset) + (i - ((core_id / (ava_cores)) * offset1)));
            if (sum < minimum[core_id]){
              // printf("now core %d and minimu is %f and i is %d and my index is %d\n", core_id, minimum[core_id],i,( offset) + (i - ((core_id / (ava_cores)) * offset1)));
              
                minimum[core_id] = sum;
                myindex[core_id] =  ( offset) + (i - ((core_id / (ava_cores)) * offset1)); //((int)(counter / 2) * 2 * w_b) + i;
                //  printf("after core %d and minimu is %f and my index is %d\n", core_id, minimum[core_id], myindex[core_id]);
              
                // location = i;
            }
          #else
            temp = (VPixel) {0, 0};
            for (int j = 0; j < M ; j+=4) 
              {
                // if (core_id== 4)
                  // printf("core id is %d i %d j %d som %f x[%d] %f\n", core_id ,i, j, SOM[ i * M + j],((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j, x[((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j] );
                // printf("i   %d j is %d som %f and x is %f \n", i, j, SOM[i * M +j], x[x_offset + j]);
                Inpv0  = *((VPixel *) &x[x_offset + j]);
                Sv0  = *((VPixel *) &SOM[ i * M + j]);

                Inpv1  = *((VPixel *) &x[x_offset + j+2]);
                Sv1  = *((VPixel *) &SOM[ i * M + j+2]);

                temp += FABS(Sv0 - Inpv0);
                temp += FABS(Sv1 - Inpv1);

              }
              sum = temp[0] + temp[1];
            // printf("temp is %f i is %d\n", sum, ( offset) + (i - ((core_id / (ava_cores)) * offset1)));
            if (sum < minimum[core_id]){
              // printf("now core %d and minimu is %f and i is %d and my index is %d\n", core_id, minimum[core_id],i,( offset) + (i - ((core_id / (ava_cores)) * offset1)));
              
                minimum[core_id] = sum;
                myindex[core_id] =  ( offset) + (i - ((core_id / (ava_cores)) * offset1)); //((int)(counter / 2) * 2 * w_b) + i;
                //  printf("after core %d and minimu is %f and my index is %d\n", core_id, minimum[core_id], myindex[core_id]);
              
                // location = i;
            }
          #endif


        #else
        temp = 0;
         for (int j = 0; j < M ; j+=2) 
            {
              // if (core_id== 4)
              //   printf("core id is %d i %d j %d som %f x[%d] %f\n", core_id ,i, j, SOM[ i * M + j],((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j, x[((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j] );
              // printf("i   %d j is %d som %f and x is %f \n", i, j, SOM[i * M +j], x[x_offset + j]);
              temp += FABS(SOM[ i * M + j] - x[x_offset + j]);
              // printf("temp %f, ",FABS(SOM[ i * M + j] - x[x_offset + j]));
              temp += FABS(SOM[ i * M + j+1] - x[x_offset + j+1]);
              // printf("temp %f, ",FABS(SOM[ i * M + j+1] - x[x_offset + j+1]));

            }
          // printf("sum %f \n", temp);
          // printf("temp is %f i is %d\n", temp, ( offset) + (i - ((core_id / (ava_cores)) * offset1)));
          if (temp < minimum[core_id]){
            // printf("now core %d and minimu is %f temp is %f and i is %d and my index is %d\n", core_id, minimum[core_id], temp,i,( offset) + (i - ((core_id / (ava_cores)) * offset1)));
            
              minimum[core_id] = temp;
              myindex[core_id] =  ( offset) + (i - ((core_id / (ava_cores)) * offset1)); //((int)(counter / 2) * 2 * w_b) + i;
              //  printf("after core %d and minimu is %f and my index is %d\n", core_id, minimum[core_id], myindex[core_id]);
            
              // location = i;
          }

      #endif
        //   location = ((int)(counter / 2) * 2 * w_b) + i;
        // }
      }

  #ifndef once_load
    if (core_id==0)
    {

      pi_cl_dma_cmd_wait(&commandA);
      #ifndef IN_ORDER
        pi_cl_dma_cmd_wait(&commandB);
      #endif
    }

    }
  #endif
 
    
    // printf("location is %d \n", location);
    // printf("%d\n", location);
  #if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif  
    
    #ifndef FP8
    for (int i = (core_id/ (ava_cores))*(ava_cores); i < ((core_id/ (ava_cores)) +1)*(ava_cores); i++)
    {
      // printf("core %d i %d \n", core_id, i);
      if (minimum[i]< minu)
      {

        minu = minimum[i];
        location = myindex[i];
        // printf("core %d i %d minimum %f my index %d new minu %f and new locat %d\n", core_id, i, minimum[i], myindex[i], minu, location);
      }
      
    }
    #else
    for (int i = (core_id/ (ava_cores))*(ava_cores); i < ((core_id/ (ava_cores)) +1)*(ava_cores); i++)
    {
      // printf("core %d i %d \n", core_id, i);
      if (minimum[i]< minu)
      {

        minu = minimum[i];
        location = myindex[i];
        // printf("core %d i %d minimum %f my index %d new minu %f and new locat %d\n", core_id, i, minimum[i], myindex[i], minu, location);
      }
      else if (minimum[i] == minu && myindex[i] < location)
      {
        minu = minimum[i];
        location = myindex[i];
        // printf(" in else if core %d i %d minimum %f my index %d new minu %f and new locat %d\n", core_id, i, minimum[i], myindex[i], minu, location);
     
      }

    }
    #endif
  

  #if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif  
  // if (core_id ==4)
    // printf("core id %d , location is %d\n", core_id, location);
  // if (core_id ==4)
  // printf("%d\n",location);
    return location;
}

/*Pixel find_BMU_value(Pixel * __restrict__ x, Pixel * __restrict__ SOM, int N, int M){

  Pixel min =min_start;

  for (int i = 0; i < N; i++) {
    Pixel temp = 0;
    for (int j = 0; j < M; j++) {

      //printf("i %d j %d som %f x %f\n", i, j, SOM[ i * M + j], x[j] );
      temp += FABS(SOM[ i * M + j] - x[j]);

      }
      // printf("temp is %f \n", temp);
      if (temp < min){
          min = temp;
      }
    }
    // printf("min is %f \n", min);
    return min;
}*/


void __attribute__ ((noinline)) update_weights(Pixel * __restrict__ x, Pixel * __restrict__ SOM_N, int N, int M, Pixel beta, int location, int s) {

int dist =0;
Pixel power=0;
#ifdef FP32
int thereshold =19;// 9223372036854775807;
#else
int thereshold = 14;
#endif

#ifdef VECTORIAL
  VPixel *output0, *output1;
  VPixel Inpv0, Inpv1, Sv0, Sv1;
  Pixel sum =0; 
  VPixel temp0,temp1;
  VPixel pow_vec;

#else
  Pixel temp = 0;
#endif
Pixel dist_func = 0.0;
Pixel dist_func1 = 0.0;
Pixel pow_temp=0;
int start =0;//0
int length = 127;
int from =0;
int extra = 0;
int adder =0;
int base =0;
int start_par = 0;
int end=0;
int ava_cores =0;
int x_offset = 0;
bool flag_begin = false;
bool flag_end = false;
pi_cl_dma_cmd_t commandB;
pi_cl_dma_cmd_t commandA;
int core_id = pi_core_id();

  // for (int j = 0; j < M; j++) 
  // {
  //   printf("x %d is %f \n",j, x[j]);
  // }

  // printf("coreid[%d] here in update ch0 \n", core_id);
 #if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif  

  #ifdef IN_ORDER
    ava_cores = NUM_CORES;
  #else
    ava_cores = NUM_CORES/2;
  #endif


   x_offset =((core_id/(ava_cores)) * 2 * inp_b * M);

  if (location <thereshold){

    // from .... N, start(0) ..location ... length
    // from .... N is done in the first loop
    start = ((core_id / (ava_cores)) * N) + 0;
    length = location + thereshold +1 ;//63+1;

    from = ((core_id / (ava_cores)) * N) + N - (thereshold - location);
    extra =(thereshold - location) ;
    flag_begin = true; 

    #ifndef once_load
      base =((core_id / (ava_cores)) * thereshold); // base address for the the extra part
      adder = ((((core_id / (ava_cores)) *3 ) +2) * thereshold);// address for the part which is greater than location
      

      #ifndef IN_ORDER
        if (core_id == (ava_cores)-1)
          {
            pi_cl_dma_cmd_2d((int) (SOM_N   + (from*M)) , (int) (SOM + base*M) , extra*M*sizeof(SOM_N[0]),
            extra*M*sizeof(SOM_N[0]),extra*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_EXT2LOC, &commandA);
            pi_cl_dma_cmd_wait(&commandA);
          }
      
        else if (core_id == (ava_cores))
          {
            pi_cl_dma_cmd_2d((int) (SOM_N + ( from*M)) , (int) (SOM + base*M) , extra*M*sizeof(SOM_N[0]),
            extra*M*sizeof(SOM_N[0]),extra*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_EXT2LOC, &commandB);
            pi_cl_dma_cmd_wait(&commandB);
          }
      #else
        if (core_id == 0)
        {
          pi_cl_dma_cmd_2d((int) (SOM_N   + (from*M)) , (int) (SOM + base*M) , extra*M*sizeof(SOM_N[0]),
          extra*M*sizeof(SOM_N[0]),extra*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_EXT2LOC, &commandA);
          pi_cl_dma_cmd_wait(&commandA);
        }
      #endif

      
    #else
      base = from;// the extra part 
      adder = ((core_id / (ava_cores)) * w_b) + 0 ; // the part > location Note: N =w_b in this case
    #endif
    // printf("core_id[%d], location is %d, from %d extra %d base %d start %d, length %d, adder %d, \n", core_id, location, from, extra, base, start, length, adder);

  }
  else if (location > N - thereshold)
  {

    //  start .... location .. length (N),  from(0) .... extra,

    //from ... extra is doen by the first loop
    start = ((core_id / (ava_cores)) * N) + location -thereshold;
    length = N - location +thereshold;;
    
    from =((core_id / (ava_cores)) * N) + 0;
    extra = thereshold - (N - location)+1;
    flag_end = true; 
  
    #ifndef once_load
      base =((core_id / (ava_cores)) * thereshold); // base address for the the extra part
      adder = ((((core_id / (ava_cores)) *3 ) +2) * thereshold);// base for the main part
      
      #ifndef IN_ORDER
      if (core_id == (ava_cores)-1)
        {
          pi_cl_dma_cmd_2d((int) (SOM_N +  (from*M)) , (int) (SOM + base*M) , extra*M*sizeof(SOM_N[0]),
          extra*M*sizeof(SOM_N[0]),extra*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_EXT2LOC, &commandA);
          pi_cl_dma_cmd_wait(&commandA);
        }
      
          else if (core_id == (ava_cores)){
              pi_cl_dma_cmd_2d((int) (SOM_N +  (from*M)) , (int) (SOM + base*M) , extra*M*sizeof(SOM_N[0]),
              extra*M*sizeof(SOM_N[0]),extra*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_EXT2LOC, &commandB);
              pi_cl_dma_cmd_wait(&commandB);
          }
      #else
        if (core_id == 0)
        {
          pi_cl_dma_cmd_2d((int) (SOM_N +  (from*M)) , (int) (SOM + base*M) , extra*M*sizeof(SOM_N[0]),
          extra*M*sizeof(SOM_N[0]),extra*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_EXT2LOC, &commandA);
          pi_cl_dma_cmd_wait(&commandA);
        }
      #endif
    #else
      base =from;
      adder = start;
    #endif
    // printf("core_id[%d], location is %d, from %d extra %d base %d start %d, length %d, adder %d, \n", core_id, location, from, extra, base, start, length, adder);

  }
  else
  {

    // start ...location... lenght 
    start= ((core_id / (ava_cores)) * N) + location - thereshold;
    length = 2 * thereshold +1;//127;
    from =0;
    extra =0;
    #ifndef once_load
        #ifdef IN_ORDER
        adder = thereshold;//because of small block size, some times we don't have 2 * thereshold size. for example, block =64 and the threshold is 38 and length =39 -> invlaid access
        #else
          adder = ((((core_id / (ava_cores)) *3 ) +2) * thereshold);// because of double buffering
        #endif
    #else
    adder = start;// correct address
    #endif
    
    // printf("core_id[%d], location is %d, start %d, length %d, adder %d, \n", core_id, location, start, length, adder);

  }
   #if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif  

  //printf(" start is %d length is %d \n", start, length);
  #ifndef once_load

      #ifndef IN_ORDER
      if (core_id == (ava_cores)-1)
        {
          pi_cl_dma_cmd_2d((int) (SOM_N  +  (start*M)) , (int) (SOM + (adder * M )) , length*M*sizeof(SOM_N[0]),
          length*M*sizeof(SOM_N[0]),length*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_EXT2LOC, &commandB);
          
        }
      
          else if (core_id == (ava_cores))
          {
              pi_cl_dma_cmd_2d((int) (SOM_N +  (start*M)) , (int) (SOM + (adder * M )) , length*M*sizeof(SOM_N[0]),
              length*M*sizeof(SOM_N[0]),length*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_EXT2LOC, &commandB);
            }
      #else
        if (core_id == 0)
        {
          pi_cl_dma_cmd_2d((int) (SOM_N  +  (start*M)) , (int) (SOM + (adder * M )) , length*M*sizeof(SOM_N[0]),
          length*M*sizeof(SOM_N[0]),length*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_EXT2LOC, &commandB);
          
        }
      #endif
  #endif
// printf("coreid[%d] here in update ch1 \n", core_id);
  #if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif
// printf("coreid[%d] here in update ch2 \n", core_id);  

// printf(" Core id %d and the location is %d \n", core_id, location);

if (flag_begin)
{

  #if NUM_CORES > 1
      int blockSize =(extra+(ava_cores)-1)/(ava_cores);
      if (core_id < (ava_cores))
        {
          start_par = core_id*blockSize;
          end = start_par + blockSize < extra? start_par + blockSize : extra;
        }
      else
        {
          start_par = (core_id - (ava_cores)) *blockSize ;
          end = start_par + blockSize < extra? start_par + blockSize : extra;
        }
  #else
   start_par = 0;
   end = extra;
  #endif

  // printf("core %d LOCATION %d base %d and i %d, extra is %d start %d end %d \n", core_id, location, base,(base + start_par) * M ,extra, start_par, end);
  for (int i =start_par; i < end; i ++)
  {
    
    dist = thereshold - i;
    //  printf("core_id[%d], location is %d, start %d, length %d, adder %d, i is %d dist is %d\n", core_id, location, start, length, adder, i, dist);

    //  printf(" dist is %d \n", dist);
    // power = 1 << dist;//(unsigned long long)1 << dist;//isinf()
    //power = powf(2,dist);


      // printf("power is %f \n", power );
    
      // pow_temp = (Pixel)(beta / power);//(pow(2.0, dist)));
      pow_temp = (-beta) *  neg_power_of_two(dist);
      //pow_temp = (-beta) *  neg_power_of_two(dist); used before removing the loop
      // printf("beta is %f power_temp is %f \n", pow_temp );
    
      // printf("core_id[%d], location is %d, i is %d dist is %d, neg power is %f beta is %f power_temp %f \n", core_id, location, i, dist,neg_power_of_two(dist), beta, pow_temp);
      #ifdef VECTORIAL
      
      #ifdef FP8
      pow_vec = (VPixel) {pow_temp, pow_temp, pow_temp,pow_temp};
        #ifndef No_for    
          for (int j = 0; j < M ; j+=8) 
        #else
          int j =0;
        #endif
            {
            // int j = 0;
              // temp0 = (VPixel){0,0};
              // temp1 = (VPixel){0,0};
              // if (core_id== 4)
                // printf("core id is %d i %d j %d som %f x[%d] %f\n", core_id ,i, j, SOM[ i * M + j],((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j, x[((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j] );
              // printf("i   %d j is %d som %f and x is %f \n", i, j, SOM[i * M +j], x[x_offset + j]);
              Inpv0  = *((VPixel *) &x[x_offset + j]);
              output0 =(VPixel *) &SOM[(base + i) * M + j];
              Sv0  = *output0;//*((VPixel *) &SOM[(base + i) * M + j]);



              Inpv1  = *((VPixel *) &x[x_offset + j+4]);
              output1 =(VPixel *) &SOM[(base + i) * M + j+4];
              Sv1  = *output1;//*((VPixel *) &SOM[(base + i) * M + j+2]);

              // pow_vec = (VPixel) {pow_temp, pow_temp};


              temp0 = pow_vec * (Sv0 - Inpv0);
              // printf("temp0 is %f temp1 is %f temp2 is %f temp3 is %f\n", temp0[0],temp0[1],temp0[2],temp0[3]);
              temp1 = pow_vec * (Sv1 - Inpv1);

              // Sv0 = Sv0 - temp0; used before removing loop
              // Sv1 = Sv1 - temp1;
              Sv0 = Sv0 + temp0;
              Sv1 = Sv1 + temp1;

              *output0 = Sv0;
              // SOM[(base + i) * M + j+1] = Sv0[1];
              *output1 = Sv1;
              // SOM[(base + i) * M + j+3] = Sv1[1];

            }
      #else // float16 or bfloat16 vectorization
        // pow_vec = (VPixel) {pow_temp, pow_temp};
        pow_vec =__builtin_pulp_floats_to_vf16(pow_temp, pow_temp);
        for (int j = 0; j < M ; j+=4)  
            {
              // temp0 = (VPixel){0,0};
              // temp1 = (VPixel){0,0};
              // if (core_id== 4)
                // printf("core id is %d i %d j %d som %f x[%d] %f\n", core_id ,i, j, SOM[ i * M + j],((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j, x[((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j] );
              // printf("i   %d j is %d som %f and x is %f \n", i, j, SOM[i * M +j], x[x_offset + j]);
              Inpv0  = *((VPixel *) &x[x_offset + j]);
              output0 =(VPixel *) &SOM[(base + i) * M + j];
              Sv0  = *output0;//*((VPixel *) &SOM[(base + i) * M + j]);



              Inpv1  = *((VPixel *) &x[x_offset + j+2]);
              output1 =(VPixel *) &SOM[(base + i) * M + j+2];
              Sv1  = *output1;//*((VPixel *) &SOM[(base + i) * M + j+2]);

              // pow_vec = (VPixel) {pow_temp, pow_temp};
              // printf("pwer %f\n", pow_vec[0]);

              temp0 = pow_vec * (Sv0 - Inpv0);
              
              temp1 = pow_vec * (Sv1 - Inpv1);

              // printf("j is %d, powe_vec is  , som [before updating] %f ,  %f , %f , %f\n",j, Sv0[0],Sv0[1],Sv1[0],Sv1[1] );
              Sv0 = Sv0 + temp0;
              Sv1 = Sv1 + temp1;
              // printf("j is %d, powe_vec is %f %f , som [after updating] %f ,  %f , %f , %f\n",j, pow_vec[0],pow_vec[1], Sv0[0],Sv0[1],Sv1[0],Sv1[1] );
              *output0 = Sv0;
              // SOM[(base + i) * M + j+1] = Sv0[1];
              *output1 = Sv1;
              // SOM[(base + i) * M + j+3] = Sv1[1];
              

            }
      #endif

      #else

      for (int j =0; j< M; j+=2)
      {
        //  printf("core_id[%d], location is %d, start %d, length %d, adder %d, i is %d dist is %d som index %d, x index %d \n", core_id, location, start, length, adder, i, dist, (base + i) * M + j,((core_id/(NUM_CORES/2)) * 2 * inp_b * M) + j);

        // printf("base is  %d address is %d \n", from, (base + i) * M + j);
        dist_func = (pow_temp * (SOM[(base + i) * M + j] - x[x_offset + j]));
        //printf("dist_func is %f \n", dist_func);
        // printf("i %d j %d , dist func is %f and SOM before %f  \n", i,j, dist_func,SOM[(base + i) * M + j]);
        SOM[(base + i) * M + j] =  SOM[(base + i )* M + j] + dist_func;
        // printf("i %d j %d after %f  \n", i,j, SOM[(base + i) * M + j]);
        dist_func1 = (pow_temp * (SOM[(base + i) * M + j+1] - x[x_offset + j+1]));
        //printf("dist_func is %f \n", dist_func);
        // printf("i %d j %d before %f  \n", i,j, SOM[i * M + j]);
        SOM[(base + i) * M + j+1] =  SOM[(base + i) * M + j+1] + dist_func1;
      }

      #endif
      // printf("flag begin finished \n");
  }
}
 

if (flag_end)
{

  #if NUM_CORES > 1
  int blockSize = (extra+(ava_cores)-1)/(ava_cores);


  if (core_id < (ava_cores))
    {
       start_par = core_id*blockSize;
       end = start_par + blockSize < extra? start_par + blockSize : extra;
    }
  else
    {
       start_par = (core_id -(ava_cores)) *blockSize ;
       end = start_par + blockSize < extra? start_par + blockSize : extra;
    }
  #else
   start_par = 0;
   end = extra;
  #endif
  // printf("flag end core %d extra is %d start %d end %d \n", core_id,extra, start_par, end);
  for (int i =start_par; i < end; i ++)
  {
    
      dist = i + (N - location);
    //  printf(" dist is %d \n", dist);
    // power = 1 << dist;//isinf()
    //power = powf(2,dist);

    // printf("power is %f \n", power );
    
    // pow_temp = (Pixel)(beta / power);//(pow(2.0, dist)));
      pow_temp = (-beta) *  neg_power_of_two(dist);
    // printf("power_temp is %f \n", pow_temp );
    


      #ifdef VECTORIAL
        #ifdef FP8
          pow_vec = (VPixel) {pow_temp, pow_temp, pow_temp,pow_temp};
        #ifndef No_for    
          for (int j = 0; j < M ; j+=8) 
        #else
          int j =0;
        #endif
                {
                  // int j = 0;
                  // temp0 = (VPixel){0,0};
                  // temp1 = (VPixel){0,0};
                  // if (core_id== 4)
                  //   printf("core id is %d i %d j %d som %f x[%d] %f\n", core_id ,i, j, SOM[ i * M + j],((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j, x[((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j] );
                  // printf("i   %d j is %d som %f and x is %f \n", i, j, SOM[i * M +j], x[x_offset + j]);
                  Inpv0  = *((VPixel *) &x[x_offset + j]);
                  output0 =(VPixel *) &SOM[(base + i) * M + j];
                  Sv0  = *output0;//*((VPixel *) &SOM[(base + i) * M + j]);

                  Inpv1  = *((VPixel *) &x[x_offset + j+4]);
                  output1 =(VPixel *) &SOM[(base + i) * M + j+4];
                  Sv1  = *output1;//*((VPixel *) &SOM[(base + i) * M + j+2]);
                  // pow_vec = (VPixel) {pow_temp, pow_temp};
                  
                  temp0 = pow_vec * (Sv0 - Inpv0);
                  // printf("temp0 is %f temp1 is %f temp2 is %f temp3 is %f\n", temp0[0],temp0[1],temp0[2],temp0[3]);
                  temp1 = pow_vec * (Sv1 - Inpv1);

              // Sv0 = Sv0 - temp0; used before removing loop
              // Sv1 = Sv1 - temp1;
                Sv0 = Sv0 + temp0;
                Sv1 = Sv1 + temp1;
                  // printf("v0 is %f v1 is %f v2 is %f v3 is %f\n", Sv0[0],Sv0[1],Sv0[2],Sv0[3]);
                  *output0 = Sv0;
                  *output1 = Sv1;
                  // SOM[(base + i) * M + j] = Sv0[0];
                  // SOM[(base + i) * M + j+1] = Sv0[1];
                  // SOM[(base + i) * M + j+2] = Sv1[0];
                  // SOM[(base + i) * M + j+3] = Sv1[1];

                }
          #else
            // pow_vec = (VPixel) {pow_temp, pow_temp};
            pow_vec =__builtin_pulp_floats_to_vf16(pow_temp, pow_temp);
            for (int j = 0; j < M ; j+=4) 
                  {
                    // temp0 = (VPixel){0,0};
                    // temp1 = (VPixel){0,0};
                    // if (core_id== 4)
                    //   printf("core id is %d i %d j %d som %f x[%d] %f\n", core_id ,i, j, SOM[ i * M + j],((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j, x[((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j] );
                    // printf("i   %d j is %d som %f and x is %f \n", i, j, SOM[i * M +j], x[x_offset + j]);
                    Inpv0  = *((VPixel *) &x[x_offset + j]);
                    output0 =(VPixel *) &SOM[(base + i) * M + j];
                    Sv0  = *output0;//*((VPixel *) &SOM[(base + i) * M + j]);

                    Inpv1  = *((VPixel *) &x[x_offset + j+2]);
                    output1 =(VPixel *) &SOM[(base + i) * M + j+2];
                    Sv1  = *output1;//*((VPixel *) &SOM[(base + i) * M + j+2]);

                    
                    
                    

                    // pow_vec = (VPixel) {pow_temp, pow_temp};


                    temp0 = pow_vec * (Sv0 - Inpv0);
                    temp1 = pow_vec * (Sv1 - Inpv1);

                    Sv0 = Sv0 + temp0;
                    Sv1 = Sv1 + temp1;
                    // printf("v0 is %f v1 is %f v2 is %f v3 is %f\n", Sv0[0],Sv0[1],Sv1[0],Sv1[1]);
                    *output0 = Sv0;
                    *output1 = Sv1;
                    // SOM[(base + i) * M + j] = Sv0[0];
                    // SOM[(base + i) * M + j+1] = Sv0[1];
                    // SOM[(base + i) * M + j+2] = Sv1[0];
                    // SOM[(base + i) * M + j+3] = Sv1[1];

                  }
          #endif

      #else

          for (int j =0; j< M; j+=2)
          {
            // printf("core_id[%d], location is %d, start %d, length %d, adder %d, i is %d dist is %d som index %d, x index %d \n", core_id, location, start, length, adder, i, dist, (base + i) * M + j,((core_id/(NUM_CORES/2)) * 2 * inp_b * M) + j);

            dist_func = (pow_temp * (SOM[(base + i) * M + j] - x[x_offset + j]));
            //printf("dist_func is %f \n", dist_func);
          //  printf("i %d j %d , dist func is %f and SOM before %f  \n", i,j, dist_func,SOM[(base + i) * M + j]);
            SOM[(base + i) * M + j] =  SOM[(base + i) * M + j] + dist_func;
            // printf("i %d j %d after %f  \n", i,j, SOM[(base + i) * M + j]);
            dist_func1 = (pow_temp * (SOM[(base + i) * M + j+1] - x[x_offset + j+1]));
            //printf("dist_func is %f \n", dist_func);
            //printf("i %d j %d before %f  \n", i,j, SOM[i * M + j]);
            SOM[(base + i) * M + j+1] =  SOM[(base + i) * M + j+1] + dist_func1;
            // printf("v0 is %f v1 is %f\n", SOM[(base + i) * M + j] - dist_func,SOM[(base + i) * M + j+1] - dist_func1);
          }
      #endif
  }
}

  #if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif

  #ifndef once_load

    #ifndef IN_ORDER
      if (core_id == (ava_cores)-1)
        {
          pi_cl_dma_cmd_wait(&commandB);
        }
      
      else if (core_id == (ava_cores))
        {
            pi_cl_dma_cmd_wait(&commandB);
          }
    #else
      if (core_id == 0)
        {
          pi_cl_dma_cmd_wait(&commandB);
        }
    #endif

#if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif

    if (flag_begin  || flag_end)
      {

        #ifndef IN_ORDER
          if (core_id == (ava_cores)-1)
            {
             pi_cl_dma_cmd_2d((int) (SOM_N  +  (from*M)) , (int) (SOM+(base*M)) , extra*M*sizeof(SOM_N[0]),
             extra*M*sizeof(SOM_N[0]),extra*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_LOC2EXT, &commandA);
          
            }
        
          else if (core_id == (ava_cores))
            {
              pi_cl_dma_cmd_2d((int) (SOM_N +  (from*M)) , (int) (SOM+(base*M)) , extra*M*sizeof(SOM_N[0]),
              extra*M*sizeof(SOM_N[0]),extra*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_LOC2EXT, &commandA);
            }
        #else
          if (core_id == 0)
            {
              pi_cl_dma_cmd_2d((int) (SOM_N  +  (from*M)) , (int) (SOM+(base*M)) , extra*M*sizeof(SOM_N[0]),
              extra*M*sizeof(SOM_N[0]),extra*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_LOC2EXT, &commandA);
          
            }
        #endif
      }
  #endif

  #if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif

  #if NUM_CORES > 1
    int blockSize = (length+(ava_cores)-1)/(ava_cores);

    if (core_id < (ava_cores))
      {
        start_par = core_id*blockSize;
        end = start_par + blockSize < length? start_par + blockSize : length;
      }
    else
      {
        start_par = (core_id -(ava_cores)) *blockSize ;
        end = start_par + blockSize < length? start_par + blockSize : length;
      }

  #else
   start_par = 0;
   end = length;
  #endif

  // printf("coreid[%d] here in update ch3 \n", core_id);
  // printf("final core %d length is %d start %d end %d \n", core_id,length, start_par, end);
  for (int i = start_par; i < end; i++) 
    {
      // dist = (div) - fabs(fabs(i - location) - div);
      if( location < thereshold){
          dist = ABS(i - location);
      }
      else{
        dist = ABS(i - thereshold);
      }
      pow_temp = (-beta) *  neg_power_of_two(dist);
      #ifdef VECTORIAL
        #ifdef FP8
        
          // pow_vec = (VPixel) {pow_temp, pow_temp, pow_temp,pow_temp};
          pow_vec =__builtin_pulp_floats_to_vf8(pow_temp, pow_temp, pow_temp,pow_temp);
          // printf("sv0 is %f sv1 is %f sv2 is %f sv3 is %f\n", pow_vec[0],pow_vec[1],pow_vec[2],pow_vec[3]);
          // printf("sv0 is %d sv1 is %d sv2 is %d sv3 is %d\n", i, x_offset,adder,M);

         #ifndef No_for    
          for (int j = 0; j < M ; j+=8) 
        #else
          int j =0;
        #endif
         
            {
               
                  // asm volatile("":::"memory");
                  Inpv0  = *((VPixel *) &x[x_offset + j]);
                  // printf("inp0 is %f inp1 is %f inp2 is %f inp3 is %f\n", Inpv0[0],Inpv0[1],Inpv0[2],Inpv0[3]);
                  output0 =(VPixel *) &SOM[(i+adder) * M + j];
                  // asm volatile("":::"memory");
                  Sv0  = *output0; //*((VPixel *) &SOM[(i+adder) * M + j]);
                  // printf("sv0 is %f sv1 is %f sv2 is %f sv3 is %f\n", Sv0[0],Sv0[1],Sv0[2],Sv0[3]);
                  
                  Inpv1  = *((VPixel *) &x[x_offset + j+4]);
                  output1 =(VPixel *) &SOM[(i+adder) * M + j+4];
                  Sv1  = *output1;//*((VPixel *) &SOM[(i+adder) * M + j+2]);
                  // printf("inp10 is %f inp11 is %f inp12 is %f inp13 is %f\n", Inpv1[0],Inpv1[1],Inpv1[2],Inpv1[3]);
                  
                  // printf("sv10 is %f sv11 is %f sv12 is %f sv13 is %f\n", Sv1[0],Sv1[1],Sv1[2],Sv1[3]);                  

                  // pow_vec = (VPixel) {pow_temp, pow_temp};

                  // asm volatile("":::"memory");
                  temp0 = pow_vec * (Sv0 - Inpv0);
                  // printf("temp0 is %f temp1 is %f temp2 is %f temp3 is %f\n", temp0[0],temp0[1],temp0[2],temp0[3]);
                  // asm volatile("":::"memory");
                  temp1 = pow_vec * (Sv1 - Inpv1);
                  // printf("temp10 is %f temp11 is %f temp12 is %f temp13 is %f\n", temp1[0],temp1[1],temp1[2],temp1[3]);
                  Sv0 = Sv0 + temp0;
                  Sv1 = Sv1 + temp1;
                  // printf("v0 is %f v1 is %f v2 is %f v3 is %f\n", Sv0[0],Sv0[1],Sv0[2],Sv0[3]);
                  // printf("v10 is %f v11 is %f v12 is %f v13 is %f\n", Sv1[0],Sv1[1],Sv1[2],Sv1[3]);
                  *output0 = Sv0;
                  *output1 = Sv1;
                  // SOM[(i+adder) * M + j] = Sv0[0];
                  // SOM[(i+adder) * M + j+1] = Sv0[1];
                  // SOM[(i+adder) * M + j+2] = Sv1[0];
                  // SOM[(i+adder) * M + j+3] = Sv1[1];
                  
              }
              
              
        #else
          // pow_vec = (VPixel) {pow_temp, pow_temp};
          pow_vec =__builtin_pulp_floats_to_vf16(pow_temp, pow_temp);
          for (int j = 0; j < M ; j+=4) 
            {
              // temp0 = (VPixel){0,0};
              // temp1 = (VPixel){0,0};
              // if (core_id== 4)
              //   printf("core id is %d i %d j %d som %f x[%d] %f\n", core_id ,i, j, SOM[ i * M + j],((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j, x[((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j] );
              // printf("i   %d j is %d som %f and x is %f \n", i, j, SOM[i * M +j], x[x_offset + j]);
              Inpv0  = *((VPixel *) &x[x_offset + j]);
              output0 =(VPixel *) &SOM[(i+adder) * M + j];
              Sv0  = *output0; //*((VPixel *) &SOM[(i+adder) * M + j]);

              Inpv1  = *((VPixel *) &x[x_offset + j+2]);
              output1 =(VPixel *) &SOM[(i+adder) * M + j+2];
              Sv1  = *output1;//*((VPixel *) &SOM[(i+adder) * M + j+2]);

              
              

              // pow_vec = (VPixel) {pow_temp, pow_temp};


              temp0 = pow_vec * (Sv0 - Inpv0);
              temp1 = pow_vec * (Sv1 - Inpv1);

              Sv0 = Sv0 + temp0;
              Sv1 = Sv1 + temp1;
              *output0 = Sv0;
              *output1 = Sv1;
              // SOM[(i+adder) * M + j] = Sv0[0];
              // SOM[(i+adder) * M + j+1] = Sv0[1];
              // SOM[(i+adder) * M + j+2] = Sv1[0];
              // SOM[(i+adder) * M + j+3] = Sv1[1];

            }
            #endif

      #else

      for (int j =0; j< M; j+=2)
      {
        // printf("base is  %d address is %d \n", base, (base + i) * M + j);
        dist_func = (pow_temp * (SOM[(i+adder) * M + j] - x[x_offset + j]));
        // printf("x is %f, x1 is %f \n",x[x_offset + j],x[x_offset + j+1]);
        // printf("s is %f, s1 is %f \n",SOM[(i+adder) * M + j],SOM[(i+adder) * M + j+1]);
        

        // if (s ==1){
      //   printf("core %d dist %d pow %f ist_func is %f \n", core_id, dist,pow_temp, dist_func);
      //  printf("i %d j %d , dist func is %f and SOM[%d] before %f  \n", i,j, dist_func,(i+adder) * M + j,SOM[(i+adder) * M + j]);
        // }
        // printf("v0 is %f v1 is %f\n", SOM[(i+adder) * M + j] - dist_func,SOM[(i+adder) * M + j+1] -(pow_temp * (SOM[(i+adder) * M + j+1] - x[x_offset + j+1])) );
        SOM[(i+adder) * M + j] =  SOM[(i+adder) * M + j] + dist_func;
        //  if (s ==1){
        // printf("i %d j %d SOM[%d] after %f  \n", i,j, (i+adder) * M + j, SOM[(i+adder) * M + j]);
        dist_func1 = (pow_temp * (SOM[(i+adder) * M + j+1] - x[x_offset + j+1]));
        // printf("temp0 %f temp1 %f\n", dist_func, dist_func1);
        //printf("dist_func is %f \n", dist_func);
        // printf("i %d j %d before %f  \n", i,j+1, SOM[(i+adder) * M + j+1]);
        SOM[(i+adder) * M + j+1] =  SOM[(i+adder) * M + j+1] + dist_func1;
        // printf("v0 is %f v1 is %f\n", SOM[(base + i) * M + j] - dist_func,SOM[(base + i) * M + j+1] - dist_func1);
        // printf("i %d j %d after %f  \n", i,j+1, SOM[(i+adder) * M + j+1]);
      }
    #endif
    }
    // printf("coreid[%d] here in update ch4 \n", core_id);

  #if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif 

  #ifndef once_load  // load and save data to L2
    if (flag_begin  || flag_end)
      {
        #ifndef IN_ORDER
          if (core_id == (ava_cores)-1)
            {
            pi_cl_dma_cmd_wait(&commandA);

            }
        
          else if (core_id == (ava_cores))
            {
              pi_cl_dma_cmd_wait(&commandA);
            }
        #else
          if (core_id == 0)
            {
              pi_cl_dma_cmd_wait(&commandA);

            }
        
        #endif
      }

    #ifndef IN_ORDER
      if (core_id == (ava_cores)-1)
          {
            pi_cl_dma_cmd_2d((int) (SOM_N  +  ((start)*M)) , (int) (SOM+ (adder * M)) , length*M*sizeof(SOM[0]),
            length*M*sizeof(SOM[0]),length*M*sizeof(SOM[0]),PI_CL_DMA_DIR_LOC2EXT, &commandA);
            pi_cl_dma_cmd_wait(&commandA);
          }
      
        else if (core_id == (ava_cores))
        {
            pi_cl_dma_cmd_2d((int) (SOM_N +  ((start)*M)) , (int) (SOM+ (adder * M)) , length*M*sizeof(SOM[0]),
            length*M*sizeof(SOM[0]),length*M*sizeof(SOM[0]),PI_CL_DMA_DIR_LOC2EXT, &commandB);
            pi_cl_dma_cmd_wait(&commandB);
        }
    #else
      if (core_id == 0)
        {
          pi_cl_dma_cmd_2d((int) (SOM_N +  ((start)*M)) , (int) (SOM+ (adder * M)) , length*M*sizeof(SOM[0]),
          length*M*sizeof(SOM[0]),length*M*sizeof(SOM[0]),PI_CL_DMA_DIR_LOC2EXT, &commandA);
          pi_cl_dma_cmd_wait(&commandA);
        }
    #endif
  #endif

  #if NUM_CORES > 1
   pi_cl_team_barrier();
  #endif 
  // printf("coreid[%d] here in update ch5 \n", core_id); 
}

// DMA : https://greenwaves-technologies.com/manuals/BUILD/PMSIS_API/html/group__ClusterDMA.html#ga80a4fad236457eb5f90b3da6ccaa6c99





void __attribute__ ((noinline)) SOM_Train(Pixel * __restrict__ Input, Pixel * __restrict__ SOM_N, int N, int M, int I, int Epoch) {

  
  pi_cl_dma_cmd_t commandA;
  pi_cl_dma_cmd_t commandB;
  Pixel beta_min=.01, decay_factor=0.99,beta=1.0;
  //int i =0;
  int location =0;
  int core_id = pi_core_id();

  // printf("Address input %x som %x  beta %x\n", INP, SOM, &beta );


  // Load inputs 
  /*#if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif 
  if (core_id == 0) // load the first input data
      {
        pi_cl_dma_cmd_2d((int) Input, (int) INP ,inp_b*M*sizeof(Input[0]),
        inp_b*M*sizeof(Input[0]),inp_b*M*sizeof(Input[0]),PI_CL_DMA_DIR_EXT2LOC, &commandA);
        pi_cl_dma_cmd_wait(&commandA);
      
        #ifndef IN_ORDER // load the second input data if we run both SOMs in parallel
              pi_cl_dma_cmd_2d((int) (Input + (I * M)), (int) (INP + (2 * inp_b * M)) ,inp_b*M*sizeof(Input[0]),
              inp_b*M*sizeof(Input[0]),inp_b*M*sizeof(Input[0]),PI_CL_DMA_DIR_EXT2LOC, &commandB);
              pi_cl_dma_cmd_wait(&commandB);
        #endif
      }
  
  #if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif */
  
// load SOM
  #ifdef once_load // load the whole SOM if we could load all the networks  
    
    #ifndef IN_ORDER // load data from both SOMs 
      if (core_id == 0)
      {
        pi_cl_dma_cmd_2d((int) SOM_N , (int) SOM ,w_b*M*sizeof(SOM_N[0]),
        w_b*M*sizeof(SOM_N[0]),w_b*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_EXT2LOC, &commandB);
        pi_cl_dma_cmd_wait(&commandB);

        pi_cl_dma_cmd_2d((int) (SOM_N + ( N * M) ), (int) (SOM + 1 * w_b * M ) ,w_b*M*sizeof(SOM_N[0]),
        w_b*M*sizeof(SOM_N[0]),w_b*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_EXT2LOC, &commandA);
        pi_cl_dma_cmd_wait(&commandA);
        }
    #else // load only the first SOM
      if (core_id == 0)
      {
        pi_cl_dma_cmd_2d((int) SOM_N , (int) SOM ,w_b*M*sizeof(SOM_N[0]),
        w_b*M*sizeof(SOM_N[0]),w_b*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_EXT2LOC, &commandB);
        pi_cl_dma_cmd_wait(&commandB);
      }
    #endif
  
  #endif

 
  #if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif

  #ifdef IN_ORDER
    for (int s =0; s < R_g; s++)
    {

      // printf("SOM %d is started\n", s);
       //IN_ORDER, we  need to load the first part of each input
      // if (core_id == 0)
      // {
      //   pi_cl_dma_cmd_2d((int) Input+ (s * I * M), (int) INP ,inp_b*M*sizeof(Input[0]),
      //   inp_b*M*sizeof(Input[0]),inp_b*M*sizeof(Input[0]),PI_CL_DMA_DIR_EXT2LOC, &commandA);
      //   pi_cl_dma_cmd_wait(&commandA);
      // }
      beta_min=.01;
      decay_factor=0.99;
      beta=1.0;
  #else
   int s = 0;
  #endif  


  for (int epoch=0; epoch < Epoch; epoch ++ )
  {

  /*#if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif 
  if (core_id == 0) // load the first input data
      {
        pi_cl_dma_cmd_2d((int) Input, (int) INP ,inp_b*M*sizeof(Input[0]),
        inp_b*M*sizeof(Input[0]),inp_b*M*sizeof(Input[0]),PI_CL_DMA_DIR_EXT2LOC, &commandA);
        pi_cl_dma_cmd_wait(&commandA);
  
      #ifndef IN_ORDER // load the second input data if we run both SOMs in parallel
        pi_cl_dma_cmd_2d((int) (Input + (I * M)), (int) (INP + (2 * inp_b * M)) ,inp_b*M*sizeof(Input[0]),
        inp_b*M*sizeof(Input[0]),inp_b*M*sizeof(Input[0]),PI_CL_DMA_DIR_EXT2LOC, &commandB);
        pi_cl_dma_cmd_wait(&commandB);
      #endif
      }
  #if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif*/ 
  

  // Load inputs 
  if (core_id == 0) // load the first input data
    {

      pi_cl_dma_cmd_2d((int) (Input + (s * I * M)), (int) INP ,inp_b*M*sizeof(Input[0]),
      inp_b*M*sizeof(Input[0]),inp_b*M*sizeof(Input[0]),PI_CL_DMA_DIR_EXT2LOC, &commandA);
      pi_cl_dma_cmd_wait(&commandA);

  #ifndef IN_ORDER // load the second input data if we run both SOMs in parallel

        pi_cl_dma_cmd_2d((int) (Input + ((s+1) * I * M)), (int) (INP + (2 * inp_b * M)) ,inp_b*M*sizeof(Input[0]),
        inp_b*M*sizeof(Input[0]),inp_b*M*sizeof(Input[0]),PI_CL_DMA_DIR_EXT2LOC, &commandB);
        pi_cl_dma_cmd_wait(&commandB);
    #endif
      }
  
 
    for (int counter =0; counter < (I / inp_b); counter++)
    {

      // printf("data for next iter from %d of input l2 put   %d l1\n", (I * M)+ ((counter +1) * inp_b*M),(((counter +1) % 2) + 2)* inp_b*M);
    if (core_id==0 && counter !=(I / inp_b)-1)
      {
        #ifndef IN_ORDER  // we need to load inputs for next iter for both SOMs
          pi_cl_dma_cmd_2d((int) (Input + ((counter +1) * inp_b*M)), (int) (INP + ((counter +1) % 2)* inp_b*M),inp_b*M*sizeof(Input[0]),
            inp_b*M*sizeof(Input[0]),inp_b*M*sizeof(Input[0]),PI_CL_DMA_DIR_EXT2LOC, &commandA);

          pi_cl_dma_cmd_2d((int) (Input + (I * M)+ ((counter +1) * inp_b*M)), (int) (INP + (((counter +1) % 2) + 2)* inp_b*M),inp_b*M*sizeof(Input[0]),
           inp_b*M*sizeof(Input[0]),inp_b*M*sizeof(Input[0]),PI_CL_DMA_DIR_EXT2LOC, &commandB);
        #else // we only need to load input for next iter for one SOM
          // printf("counter is %d s is %d data for next iter from %d of input l2 put   %d l1\n",counter, s, (s*I * M)+ ((counter +1) * inp_b*M),(((counter +1) % 2))* inp_b*M);
          pi_cl_dma_cmd_2d((int) (Input + (s * I * M)+ ((counter +1) * inp_b*M)), (int) (INP + (((counter +1) % 2) )* inp_b*M),inp_b*M*sizeof(Input[0]),
           inp_b*M*sizeof(Input[0]),inp_b*M*sizeof(Input[0]),PI_CL_DMA_DIR_EXT2LOC, &commandB);
        #endif
      }

      /*#if Epoch > 1
      else if (counter == (I / inp_b)-1 && core_id==0 && epoch < (Epoch-1) ) //  input of the current SOMs for next epoch ;
      {
                // printf("next epoch counter is %d s is %d data for next iter from %d of input l2 put   %d l1\n",counter, s, (s*I * M)+ ((counter +1) * inp_b*M),(((counter +1) % 2))* inp_b*M);
          #ifndef IN_ORDER  // we need to load inputs for next iter for both SOMs
            pi_cl_dma_cmd_2d((int) (Input ), (int) (INP ),inp_b*M*sizeof(Input[0]),
              inp_b*M*sizeof(Input[0]),inp_b*M*sizeof(Input[0]),PI_CL_DMA_DIR_EXT2LOC, &commandA);

            pi_cl_dma_cmd_2d((int) (Input + (I * M)), (int) (INP + 2* inp_b*M),inp_b*M*sizeof(Input[0]),
            inp_b*M*sizeof(Input[0]),inp_b*M*sizeof(Input[0]),PI_CL_DMA_DIR_EXT2LOC, &commandB);
          #else // we only need to load input for next iter for one SOM
            // printf("counter is %d s is %d data for next iter from %d of input l2 put   %d l1\n",counter, s, (s*I * M)+ ((counter +1) * inp_b*M),(((counter +1) % 2))* inp_b*M);
            pi_cl_dma_cmd_2d((int) (Input + (s * I * M)), (int) (INP),inp_b*M*sizeof(Input[0]),
            inp_b*M*sizeof(Input[0]),inp_b*M*sizeof(Input[0]),PI_CL_DMA_DIR_EXT2LOC, &commandB);
          #endif
      }
      #endif
    

      #ifdef IN_ORDER
      else if (counter == (I / inp_b)-1 && core_id==0 && epoch == (Epoch-1) && s < (R_g-1) ) //  input of next SOM ;
      {
              // printf("next epoch counter is %d s is %d data for next iter from %d of input l2 put   %d l1\n",counter, s, (s*I * M)+ ((counter +1) * inp_b*M),(((counter +1) % 2))* inp_b*M);

        pi_cl_dma_cmd_2d((int) (Input + ( (s+1) * I * M) + (0 * inp_b)), (int) (INP ),inp_b*M*sizeof(Input[0]),
        inp_b*M*sizeof(Input[0]),inp_b*M*sizeof(Input[0]),PI_CL_DMA_DIR_EXT2LOC, &commandA);
      }
      #endif*/


      //printf("Counter is %d loop from %d to %d \n", counter, (counter %2) * inp_b, ((counter %2 ) +1) * inp_b);
      for ( int i = (counter %2) * inp_b; i < ((counter %2 ) +1) * inp_b; i++) 
      {

          // printf("i is %d and access to input %d  \n", i, (i * M));
          location = find_BMU_index((INP + (i * M)), SOM_N+ (s * N * M), N, M,s);

          update_weights(INP+(i * M), SOM_N+ (s * N * M), N, M, beta, location, s );
 
          // printf("out of the update %d \n", i);
          beta = beta * decay_factor;
          if (beta < beta_min)
          {
            beta =  beta_min;
          }

          
      }


      //if (core_id==0 && counter !=(I / inp_b)-1) 
      // #if Epoch > 1
      //   if (core_id==0 && (epoch < (Epoch-1) || counter !=(I / inp_b)-1) ) 
      // #else
        if (core_id==0 && counter !=(I / inp_b)-1) 
      //#endif
      //if (core_id==0 && counter !=(I / inp_b)-1) 
      {

        #ifndef IN_ORDER
          pi_cl_dma_cmd_wait(&commandA);
          pi_cl_dma_cmd_wait(&commandB);
        #else
          pi_cl_dma_cmd_wait(&commandB);
        #endif

      } 
        #if NUM_CORES > 1
        pi_cl_team_barrier();
        #endif

      /*else if (core_id==0 && counter ==(I / inp_b)-1 && epoch == (Epoch-1)) 
      {

        #ifndef IN_ORDER
          pi_cl_dma_cmd_wait(&commandA);
          pi_cl_dma_cmd_wait(&commandB);
        #else
          pi_cl_dma_cmd_wait(&commandB);
        #endif

      } */
      
      /*#ifdef IN_ORDER 
      else if (counter == (I / inp_b)-1 && core_id==0  && s < (R_g-1)) // input of next SOM ; to do: modify it to support next epoch of current som
      {
        pi_cl_dma_cmd_wait(&commandA);
      }
      #endif*/

    } //end of counter
      // #if NUM_CORES > 1
      //   pi_cl_team_barrier();
      // #endif

  } //end of epoch



  #ifdef once_load
      #ifndef IN_ORDER // we have two SOMs in the L1 and we need to move them to L2
        if (core_id==0)
        {
          pi_cl_dma_cmd_2d((int) SOM_N , (int) SOM ,2 * w_b*M*sizeof(SOM_N[0]),
          w_b*M*sizeof(SOM_N[0]),w_b*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_LOC2EXT, &commandB);
          pi_cl_dma_cmd_wait(&commandB);
        }
      #else // we need to move one SOM to L2 and load the next one to L1
        // first move the current SOM to L2
        if (core_id==0)
          {
            pi_cl_dma_cmd_2d((int) (SOM_N+ (s * N * M)) , (int) SOM , w_b*M*sizeof(SOM_N[0]),
            w_b*M*sizeof(SOM_N[0]),w_b*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_LOC2EXT, &commandA);
            pi_cl_dma_cmd_wait(&commandA);
            // move the next SOM to L1
            if (s != R_g-1){
                pi_cl_dma_cmd_2d((int) (SOM_N + ( (s+1) * N * M) ), (int) (SOM ) ,w_b*M*sizeof(SOM_N[0]),
                w_b*M*sizeof(SOM_N[0]),w_b*M*sizeof(SOM_N[0]),PI_CL_DMA_DIR_EXT2LOC, &commandA);
                pi_cl_dma_cmd_wait(&commandA);
            }
          }
      #endif
  #endif

  #ifdef IN_ORDER

    // #if NUM_CORES > 1
    // pi_cl_team_barrier();
    // #endif
    } // for on s (number of soms)
  #endif

  #if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif
}
/*Pixel __attribute__ ((noinline)) SOM_Test(Pixel * __restrict__ Input, Pixel * __restrict__ SOM, int N, int M, int K) {

  Pixel score =0;

  for (int i = 0; i < K; i++) {
    
    score += find_BMU_value((Input + (i * M)), SOM, N, M);
    // printf("score is %f\n", score);
    
  }


  #if NUM_CORES > 1
  pi_cl_team_barrier();
  #endif
  return score;
}*/


/*static void pi_cl_dma_cmd_2d	(	uint32_t 	ext,
uint32_t 	loc,
uint32_t 	size,
uint32_t 	stride,
uint32_t 	length,
pi_cl_dma_dir_e 	dir,
pi_cl_dma_cmd_t * 	cmd 
)	
This enqueues a 2D DMA memory transfer (rectangle) with simple completion based on transfer identifier.

Parameters
ext	Address in the external memory where to access the data.
loc	Address in the cluster memory where to access the data.
size	Number of bytes to be transferred.
stride	2D stride, which is the number of bytes which are added to the beginning of the current line to switch to the next one.
length	2D length, which is the number of transferred bytes after which the DMA will switch to the next line.
dir	Direction of the transfer. If it is PI_CL_DMA_DIR_EXT2LOC, the transfer is loading data from external memory and storing to cluster memory. If it is PI_CL_DMA_DIR_LOC2EXT, it is the opposite.
cmd	A pointer to the structure for the copy. This can be used with pi_cl_dma_wait to wait for the completion of this transfer.
*/


#else

int find_BMU_index(Pixel * __restrict__ x, Pixel * __restrict__ SOM, int N, int M){


  #ifdef VECTORIAL
  VPixel temp;
  VPixel Inpv0, Inpv1, Sv0, Sv1;
  Pixel sum =0; 
  #else
  Pixel temp = 0;
  #endif
  int location =0;
  Pixel minim = 448;
  // if (core_id ==0)
    // for (int j = 0; j < M; j++) 
    // {
    //   printf("x %d is %f \n",j, x[j]);
    // }

 
  // if (s ==1)
  // printf("snm %d  \n",j, (s * N * M))

      // printf("core id is %d start %d end %d \n", core_id, start_par, end);

      // printf("index function core %d start %d end %d \n", core_id, start_par, end);
      //for (int i = ((counter %2) * w_b) + start_par; i < (((counter %2 ) +1) * w_b; i++) 
      for (int i = 0; i < N; i++)
      {

        #ifdef VECTORIAL
          temp = (VPixel) {0, 0};
          for (int j = 0; j < M ; j+=4) 
            {
              // if (core_id== 4)
                // printf("core id is %d i %d j %d som %f x[%d] %f\n", core_id ,i, j, SOM[ i * M + j],((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j, x[((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j] );
              // printf("i   %d j is %d som %f and x is %f \n", i, j, SOM[i * M +j], x[x_offset + j]);
              Inpv0  = *((VPixel *) &x[j]);
              Sv0  = *((VPixel *) &SOM[ i * M + j]);

              Inpv1  = *((VPixel *) &x[j+2]);
              Sv1  = *((VPixel *) &SOM[ i * M + j+2]);

              temp += FABS(Sv0 - Inpv0);
              temp += FABS(Sv1 - Inpv1);

            }
            sum = temp[0] + temp[1];
          // printf("temp is %f i is %d\n", sum, ( offset) + (i - ((core_id / (ava_cores)) * offset1)));
          if (sum < minim){
            // printf("now core %d and minimu is %f and i is %d and my index is %d\n", core_id, minimum[core_id],i,( offset) + (i - ((core_id / (ava_cores)) * offset1)));
            
              minim = sum;
              location =i; //((int)(counter / 2) * 2 * w_b) + i;
              //  printf("after core %d and minimu is %f and my index is %d\n", core_id, minimum[core_id], myindex[core_id]);
            
              // location = i;
          }

        #else
        temp = 0;
         for (int j = 0; j < M ; j+=2) 
            {
              // if (core_id== 4)
              //   printf("core id is %d i %d j %d som %f x[%d] %f\n", core_id ,i, j, SOM[ i * M + j],((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j, x[((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j] );
              // printf("som %f and x is %f diff is %f \n", SOM[i * M +j], x[j], FABS(SOM[ i * M + j] - x[j]));
              temp += FABS(SOM[ i * M + j] - x[j]);
              // printf("temp is %f i is %d \n", temp, i);
              // printf("diff is %f j %d \n", FABS(SOM[ i * M + j] - x[j]), j);
              // printf("som %f and x is %f diff is %f \n", SOM[i * M +j+1], x[j+1], FABS(SOM[ i * M + j+1] - x[j+1]));
              
              // printf("diff is %f j %d \n", FABS(SOM[ i * M + j+1] - x[j+1]), j);
              temp += FABS(SOM[ i * M + j+1] - x[j+1]);
              // printf("temp is %f i is %d\n", temp, i);

            }
          // printf("temp is %f\n", temp);
          if (temp < minim){
            // printf("now core %d and minimu is %f and i is %d and my index is %d\n", core_id, minimum[core_id],i,( offset) + (i - ((core_id / (ava_cores)) * offset1)));
            
              minim= temp;
              location =i; //((int)(counter / 2) * 2 * w_b) + i;
              //  printf("after core %d and minimu is %f and my index is %d\n", core_id, minimum[core_id], myindex[core_id]);
            
              // location = i;
          }

      #endif
        //   location = ((int)(counter / 2) * 2 * w_b) + i;
        // }
      }
  // if (core_id ==4)
  // printf("location is%d\n", location);
    return location;
}

/*Pixel find_BMU_value(Pixel * __restrict__ x, Pixel * __restrict__ SOM, int N, int M){

  Pixel min =min_start;

  for (int i = 0; i < N; i++) {
    Pixel temp = 0;
    for (int j = 0; j < M; j++) {

      //printf("i %d j %d som %f x %f\n", i, j, SOM[ i * M + j], x[j] );
      temp += FABS(SOM[ i * M + j] - x[j]);

      }
      // printf("temp is %f \n", temp);
      if (temp < min){
          min = temp;
      }
    }
    // printf("min is %f \n", min);
    return min;
}*/


void __attribute__ ((noinline)) update_weights(Pixel * __restrict__ x, Pixel * __restrict__ SOM, int N, int M, Pixel beta, int location) {

Pixel div = (Pixel)(N / 2);
// printf("div is %f \n", div);
int dist =0;
Pixel power=0;
#ifdef FP32
int thereshold =19;// 9223372036854775807;
#else
int thereshold = 14;
#endif

#ifdef VECTORIAL
  VPixel *output0, *output1;
  VPixel Inpv0, Inpv1, Sv0, Sv1;
  Pixel sum =0; 
  VPixel pow_vec, temp0,temp1;

#else
  Pixel temp = 0;
#endif
Pixel dist_func = 0.0;
Pixel dist_func1 = 0.0;
Pixel pow_temp=0;
int start =0;//0
int length = 127;
int from =0;
int extra = 0;

if (location <thereshold){

    // from .... N, start(0) ..location ... length
    // from .... N is done in the first loop
    start =  0;
    length = location + thereshold +1 ;

    from = N - (thereshold - location);
    extra =N; //(thereshold - location) ;
    // printf("begin location is %d,from %d extra %d start %d,length %d\n", location, from, extra, start, length);

  for (int i =from; i < extra; i++)
  {
    
    dist = location + N - i;
    //  printf(" location is %d, start %d, length %d, i is %d dist is %d\n", location, start, length, i, dist);

    //  printf(" dist is %d \n", dist);
    // power = 1 << dist;//(unsigned long long)1 << dist;//isinf()
    //power = powf(2,dist);


      // printf("power is %f \n", power );
    
      // pow_temp = (Pixel)(beta / power);//(pow(2.0, dist)));
      pow_temp = beta *  neg_power_of_two(dist);
      // printf("beta is %f power_temp is %f \n", pow_temp );
    

      #ifdef VECTORIAL
      pow_vec = (VPixel) {pow_temp, pow_temp};
       for (int j = 0; j < M ; j+=4) 
            {
              // temp0 = (VPixel){0,0};
              // temp1 = (VPixel){0,0};
              // if (core_id== 4)
              //   printf("core id is %d i %d j %d som %f x[%d] %f\n", core_id ,i, j, SOM[ i * M + j],((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j, x[((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j] );
              // printf("i   %d j is %d som %f and x is %f \n", i, j, SOM[i * M +j], x[x_offset + j]);
              Inpv0  = *((VPixel *) &x[j]);
              output0 =(VPixel *) &SOM[i * M + j];
              Sv0  = *output0;//*((VPixel *) &SOM[(base + i) * M + j]);

              Inpv1  = *((VPixel *) &x[j+2]);
              output1 =(VPixel *) &SOM[ i* M + j+2];
              Sv1  = *output1;//*((VPixel *) &SOM[(base + i) * M + j+2]);

              // pow_vec = (VPixel) {pow_temp, pow_temp};


              temp0 = pow_vec * (Sv0 - Inpv0);
              temp1 = pow_vec * (Sv1 - Inpv1);

              Sv0 = Sv0 - temp0;
              Sv1 = Sv1 - temp1;

              *output0 = Sv0;
              // SOM[(base + i) * M + j+1] = Sv0[1];
              *output1 = Sv1;
              // SOM[(base + i) * M + j+3] = Sv1[1];

            }

      #else

      for (int j =0; j< M; j+=2)
      {
        //  printf("core_id[%d], location is %d, start %d, length %d, adder %d, i is %d dist is %d som index %d, x index %d \n", core_id, location, start, length, adder, i, dist, (base + i) * M + j,((core_id/(NUM_CORES/2)) * 2 * inp_b * M) + j);

        // printf("base is  %d address is %d \n", from, (base + i) * M + j);
        dist_func = (pow_temp * (SOM[ i * M + j] - x[ j]));
        //printf("dist_func is %f \n", dist_func);
        // printf("i %d j %d , dist func is %f and SOM before %f  \n", i,j, dist_func,SOM[(base + i) * M + j]);
        SOM[i * M + j] =  SOM[ i * M + j] - dist_func;
        // printf("i %d j %d after %f  \n", i,j, SOM[(base + i) * M + j]);
        dist_func1 = (pow_temp * (SOM[ i * M + j+1] - x[j+1]));
        //printf("dist_func is %f \n", dist_func);
        // printf("i %d j %d before %f  \n", i,j, SOM[i * M + j]);
        SOM[ i * M + j+1] =  SOM[ i * M + j+1] - dist_func1;
      }

      #endif
      // printf("flag begin finished \n");
  }

  }
  else if (location > N - thereshold)
  {

    //  start .... location .. length (N),  from(0) .... extra,

    //from ... extra is doen by the first loop
    start = location -thereshold;
    length = N;
    
    from = 0;
    extra = thereshold - (N - location)+1;
    // printf("core_id[%d], location is %d, from %d extra %d base %d start %d, length %d, adder %d, \n", core_id, location, from, extra, base, start, length, adder);
  // printf("end location is %d,from %d extra %d start %d,length %d\n", location, from, extra, start, length);
for (int i =from; i < extra; i++)
  {
    
      dist = i + (N - location);
    //  printf(" dist is %d \n", dist);
    // power = 1 << dist;//isinf()
    //power = powf(2,dist);

    // printf("power is %f \n", power );
    
    // pow_temp = (Pixel)(beta / power);//(pow(2.0, dist)));
      pow_temp = beta *  neg_power_of_two(dist);
    // printf("power_temp is %f \n", pow_temp );
    


      #ifdef VECTORIAL
      pow_vec = (VPixel) {pow_temp, pow_temp};
          for (int j = 0; j < M ; j+=4) 
                {
                  // temp0 = (VPixel){0,0};
                  // temp1 = (VPixel){0,0};
                  // if (core_id== 4)
                  //   printf("core id is %d i %d j %d som %f x[%d] %f\n", core_id ,i, j, SOM[ i * M + j],((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j, x[((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j] );
                  // printf("i   %d j is %d som %f and x is %f \n", i, j, SOM[i * M +j], x[x_offset + j]);
                  Inpv0  = *((VPixel *) &x[ j]);
                  output0 =(VPixel *) &SOM[ i * M + j];
                  Sv0  = *output0;//*((VPixel *) &SOM[(base + i) * M + j]);

                  Inpv1  = *((VPixel *) &x[ j+2]);
                  output1 =(VPixel *) &SOM[ i * M + j+2];
                  Sv1  = *output1;//*((VPixel *) &SOM[(base + i) * M + j+2]);
                
                  // pow_vec = (VPixel) {pow_temp, pow_temp};

                  temp0 = pow_vec * (Sv0 - Inpv0);
                  temp1 = pow_vec * (Sv1 - Inpv1);

                  Sv0 = Sv0 - temp0;
                  Sv1 = Sv1 - temp1;

                  *output0 = Sv0;
                  *output1 = Sv1;
                  // SOM[(base + i) * M + j] = Sv0[0];
                  // SOM[(base + i) * M + j+1] = Sv0[1];
                  // SOM[(base + i) * M + j+2] = Sv1[0];
                  // SOM[(base + i) * M + j+3] = Sv1[1];

                }

      #else

          for (int j =0; j< M; j+=2)
          {
            // printf("location is %d, start %d, length %d, i is %d dist is %d som index %d, x index %d \n", location, start, length, i, dist, (base + i) * M + j,((core_id/(NUM_CORES/2)) * 2 * inp_b * M) + j);

            dist_func = (pow_temp * (SOM[ i * M + j] - x[ j]));
            //printf("dist_func is %f \n", dist_func);
          //  printf("i %d j %d , dist func is %f and SOM before %f  \n", i,j, dist_func,SOM[(i) * M + j]);
            SOM[ i * M + j] =  SOM[ i * M + j] - dist_func;
            // printf("i %d j %d after %f  \n", i,j, SOM[( i) * M + j]);
            dist_func1 = (pow_temp * (SOM[ i * M + j+1] - x[ j+1]));
            //printf("dist_func is %f \n", dist_func);
            //printf("i %d j %d before %f  \n", i,j, SOM[i * M + j]);
            SOM[ i * M + j+1] =  SOM[ i * M + j+1] - dist_func1;
          }
      #endif
  }

  }
  else
  {
    // start ...location... lenght 
    start= location - thereshold;
    length = location + thereshold +1;//2 * thereshold +1;//127;
  // printf("midd location is %d,start %d,length %d\n", location, start, length);
    // printf("core_id[%d], location is %d, start %d, length %d, adder %d, \n", core_id, location, start, length, adder);

  }

  // printf("coreid[%d] here in update ch3 \n", core_id);
  // printf("final  start %d length %d \n",start, length);
  for (int i = start; i < length; i++) 
    {
      //dist = (div) - fabs(fabs(i - location) - div);
      // printf("location isssssssss  %d\n", location);
      // if( location < thereshold){
          dist = ABS(i - location);
      // }
      // else{
      //   dist = ABS(i - thereshold);
      // }
      
      // printf("dist is %d \n", dist);
      // printf("%d \n", dist);
      // if ( dist > thereshold)
      // {
      //   continue;
      // }
      // power = 1 << dist;//isinf()
    //power = powf(2,dist);


      // printf("power is %f \n", power );
    
      // pow_temp = (Pixel)(beta / power);//(pow(2.0, dist)));

      pow_temp = beta *  neg_power_of_two(dist);
    // printf("power_temp is %f \n", pow_temp );
    
      #ifdef VECTORIAL
      pow_vec = (VPixel) {pow_temp, pow_temp};
       for (int j = 0; j < M ; j+=4) 
            {
              // temp0 = (VPixel){0,0};
              // temp1 = (VPixel){0,0};
              // if (core_id== 4)
              //   printf("core id is %d i %d j %d som %f x[%d] %f\n", core_id ,i, j, SOM[ i * M + j],((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j, x[((core_id/(NUM_CORES/2)) * 2 * inp_b* M) + j] );
              // printf("i   %d j is %d som %f and x is %f \n", i, j, SOM[i * M +j], x[x_offset + j]);
              Inpv0  = *((VPixel *) &x[j]);
              output0 =(VPixel *) &SOM[i* M + j];
              Sv0  = *output0; //*((VPixel *) &SOM[(i+adder) * M + j]);

              Inpv1  = *((VPixel *) &x[ j+2]);
              output1 =(VPixel *) &SOM[i * M + j+2];
              Sv1  = *output1;//*((VPixel *) &SOM[(i+adder) * M + j+2]);

              // pow_vec = (VPixel) {pow_temp, pow_temp};

              temp0 = pow_vec * (Sv0 - Inpv0);
              temp1 = pow_vec * (Sv1 - Inpv1);

              Sv0 = Sv0 - temp0;
              Sv1 = Sv1 - temp1;
              *output0 = Sv0;
              *output1 = Sv1;
              // SOM[(i+adder) * M + j] = Sv0[0];
              // SOM[(i+adder) * M + j+1] = Sv0[1];
              // SOM[(i+adder) * M + j+2] = Sv1[0];
              // SOM[(i+adder) * M + j+3] = Sv1[1];

            }

      #else
      for (int j =0; j< M; j+=2)
      {
        // printf("base is  %d address is %d \n", base, (base + i) * M + j);
        dist_func = (pow_temp * (SOM[i * M + j] - x[j]));
        // if (s ==1){
      //   printf("dist %d pow %f ist_func is %f \n", dist,pow_temp, dist_func);
      //  printf("i %d j %d , dist func is %f and SOM before %f  \n", i,j, dist_func,SOM[i * M + j]);
        // }

        SOM[i * M + j] =  SOM[i * M + j] - dist_func;
        //  if (s ==1){
        // printf("i %d j %d after %f \n", i, j, SOM[i * M + j]);
        dist_func1 = (pow_temp * (SOM[i* M + j+1] - x[j+1]));
        //printf("dist_func is %f \n", dist_func);
        //printf("i %d j %d before %f  \n", i,j, SOM[i * M + j]);
        SOM[i* M + j+1] =  SOM[i* M + j+1] - dist_func1;
        //printf("i %d j %d after %f  \n", i,j, SOM[i * M + j]);
      }
    #endif
    }
    // printf("coreid[%d] here in update ch4 \n", core_id);
  // printf("coreid[%d] here in update ch5 \n", core_id); 
}

// DMA : https://greenwaves-technologies.com/manuals/BUILD/PMSIS_API/html/group__ClusterDMA.html#ga80a4fad236457eb5f90b3da6ccaa6c99





void __attribute__ ((noinline)) SOM_Train(Pixel * __restrict__ Input, Pixel * __restrict__ SOM_N, int N, int M, int I, int Epoch) 
{
  Pixel beta_min=.01, decay_factor=0.99,beta=1.0;
  //int i =0;
  int location =0;
  for (int s =0; s < R_g; s++)
    {
      beta_min=.01;
      decay_factor=0.99;
      beta=1.0;

    for (int epoch=0; epoch < Epoch; epoch ++ )
    {
 
      for (int i =0; i < I ; i++)
       {
      //     printf("i is %d and access to input %d  \n", i, (i * M)+ (s * I * M));
      //     printf("i is %d and access to som %d  \n", i, (s * N * M));
          location = find_BMU_index((Input + (i * M) + (s * I * M)), SOM_N+ (s * N * M), N, M);
  
          update_weights(Input+(i * M)+ (s * I * M), SOM_N+ (s * N * M), N, M, beta, location);

          // printf("out of the update %d \n", i);
          beta = beta * decay_factor;
          if (beta < beta_min)
          {
            beta =  beta_min;
          }

          
      }
    }   
}
}
#endif