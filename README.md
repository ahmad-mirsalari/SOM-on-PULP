# SOM

run the C code on cluster:
 make clean all run stats=1 check=1 w_block=128 i_block=32  cores=1 fmt=FP16 verbose=1  IN_ORDER=1 

 
on FABRIC:
make clean all run check=1 stats=1  fmt=FP16 verbose=1 FABRIC=1 IN_ORDER=1



v1 -> .just one som, and once load v2-> two som(at least two cores), and once load v3 -> two som(sequential code), once load is not working v4 -> two som(sequential code), once load is working, epoch>1 is working also, fp16 is okay, v5 -> add vectorization v6 -> optimize the for loop v7 -> remove the division v8 -> working on a copy of v5 and remove the division (because in V6, we couldn't use barrier for half of the cores)

I am using v8(from kth floder in my APROPOS project) on SOM github. If a file is missed on Github, refer to the kth/v8 folder.

gtkwave

get the trace file for fp32 also, and then compare the result with fp16vec also, check the new version of fp16 vec to see if the core 1 is activated in the last period or not

run the golden model:

python main_soms.py --I=10 --T=10 --R=2 --S=8 --N=12 --float_type=FP8,FP8,FP8 --MAC_flag=false --vec_flag=false

run the c Code:

make clean all run stats=1 check=1 w_block=512 i_block=32 cores=2 fmt=FP16 vec=1

