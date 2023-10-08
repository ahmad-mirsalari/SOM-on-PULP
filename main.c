// License, Version 0.51 (the "License"); you may not use this file except in
// or agreed to in writing, software, hardware and materials distributed under
// this License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "pmsis.h"
#include "config.h"
#include "stats.h"

#include <stdio.h>
#include <stdint.h>
#include <limits.h> /* for CHAR_BIT */

#include <math.h>
#include "data.h"


#define STACK_SIZE 1024
PI_L2 float  THR= 0.002f;
// End of computation
int done = 0;

PI_L2 int err = 0;
Pixel score =0;

#ifndef FP8
double __extendohfdf2(float16alt value)
{
  float result;
  __asm__ __volatile__ ("fcvt.s.ah %0, %1": "=f"(result): "f"(value) :);
  return (double) result;
}

double __extendhfdf2(float16 value)
{
  float result;
  __asm__ __volatile__ ("fcvt.s.h %0, %1": "=f"(result): "f"(value) :);
  return (double) result;
}
#endif
int __attribute ((noinline)) check_result(Pixel * __restrict__ result, Pixel score) {


  if(pi_core_id() == 0) {
    float diff;
    

   float f = 3.14159f;


    for (int i = 0; i < (R * N*M); i++) {
      diff = fabs(result[i] - ref[i]);
      if(diff > THR) {
        err++;
      #ifdef VERBOSE
      printf("Error at index %d:\t refrence %f\t output %f\t error %.4f\n", i, ref[i], result[i], diff);
      #endif
      
      }
      #ifdef PRINT_RESULTS
      printf("index %d:\t refrence %f\t output %f\t error %.5f\n", i, ref[i], result[i], diff);
      if (i == (N * M) -1 )
        printf("Next SOM \n");
      #endif

    
     }




    // if (score == score_ref)
    // {
    //   printf("FP32 TEST PASSED score_ref %f and score of C code is %f \n", score_ref, score);
    // }
    // else{
    //   printf("FP32 TEST FAILED!!!! score_ref %f and score of C code is %f \n", score_ref, score);
    // }
  
    if(err != 0)
      #ifndef VECTORIAL
        #ifdef FP32
          printf("FP32 TEST FAILED with %d errors!!\n", err);
        #elif defined(FP16)
          printf("FP16 TEST FAILED with %d errors!!\n", err);
        #elif defined(FP16ALT)
          printf("FP16ALT TEST FAILED with %d errors!!\n", err);
        #elif defined(FP8)
          printf("FP8 TEST FAILED with %d errors!!\n", err);
        #endif
      #else
        #ifdef FP16
          printf("FP16 VEC TEST FAILED with %d errors!!\n", err);
        #elif defined (FP16ALT)
          printf("FP16ALT VEC TEST FAILED with %d errors!!\n", err);
        #elif defined (FP8)
          printf("FP8 VEC TEST FAILED with %d errors!!\n", err);
        #endif
      #endif
      
    else
      #ifndef VECTORIAL
        #ifdef FP32
          printf("FP32 TEST PASSED!!\n");
        #elif defined(FP16)
          printf("FP16 TEST PASSED!!\n");
        #elif defined(FP16ALT)
          printf("FP16ALT TEST PASSED!!\n");
        #elif defined(FP8)
          printf("FP8 TEST PASSED!!\n");
        #endif
      #else
        #ifdef FP16
          printf("FP16 VEC TEST PASSED!!\n");
        #elif defined(FP16ALT)
          printf("FP16ALT VEC TEST PASSED!!\n");
        #elif defined(FP8)
          printf("FP8 VEC TEST PASSED!!\n");
        #endif
      #endif

    return err;
  }
  #ifndef FABRIC
    pi_cl_team_barrier();
  #endif
}

void main_fn(){

 
  #ifndef FABRIC
    pi_cl_team_barrier();
  #endif
  
  #ifdef STATS
  INIT_STATS();

  PRE_START_STATS();
  START_STATS();
  #endif
  
  SOM_Train(input_train, SOM_N, N, M, I, Epoch);
  //score = SOM_Test(input_test, SOM_N, N, M, K);

  #ifdef STATS
  STOP_STATS();
  #endif

  #ifdef CHECK
  check_result(SOM_N, 0);
  #endif
};

#ifndef FABRIC
static int cluster_entry(void *arg)
{
  pi_cl_team_fork(NUM_CORES, main_fn, (void *)0x0);

  return 0;
}

static void end_of_call(void *arg)
{
  done = 1;
}

static void exec_cluster_stub(void *arg)
{
  int *desc = arg;
  int (*entry)() = (int (*)())desc[0];
  desc[1] = entry();
}

static int exec_cluster(int (*cluster_entry)())
{
  int desc[2] = { (int)cluster_entry, 0 };

  struct pi_device cluster_dev;
  struct pi_cluster_conf conf;
  struct pi_cluster_task cluster_task;

  pi_cluster_conf_init(&conf);

  pi_open_from_conf(&cluster_dev, &conf);

  pi_cluster_open(&cluster_dev);

  pi_cluster_task(&cluster_task, exec_cluster_stub, desc);
      // [OPTIONAL] specify the stack size for the task
  cluster_task.stack_size = STACK_SIZE;
  cluster_task.slave_stack_size = STACK_SIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

  pi_cluster_close(&cluster_dev);

  return desc[1];
}
#endif

int main()
{

  #ifdef FABRIC
      main_fn();
  #else
    if (exec_cluster(cluster_entry))
      return -1;
  #endif
  return err;
}
