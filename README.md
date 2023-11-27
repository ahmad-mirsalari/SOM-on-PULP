# Optimizing Self-Organizing Maps for Bacterial Genome Identification on Parallel Ultra-Low-Power Platforms

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




# TransLib
TransLib, an open-source kernel library based on transprecision computing principles, provides knobs to exploit different FP data types (i.e., float, float16, and bfloat16), also considering the trade-off between homogeneous and mixed-precision solutions. 

We demonstrate the capabilities of the proposed library on PULP, a 32-bit microcontroller (MCU) coupled with a parallel, programmable accelerator. We performed a comparative analysis of seven standard kernels used in IoT end-node processing to evaluate our proposed method.

## Reference
This library is based on the research outlined in the following paper:

- Mirsalari, S.A., Tagliavini, G., Rossi, D., and Benini, L., "TransLib: A Library to Explore Transprecision Floating-Point Arithmetic on Multi-Core IoT End-Nodes", 2023 Design, Automation & Test in Europe Conference & Exhibition (DATE), 2023, [Link to the paper](https://ieeexplore.ieee.org/abstract/document/10136916)

If you find this library useful in your research, please consider citing the paper:

> ```
> @INPROCEEDINGS{10136916,
>   author={Mirsalari, Seyed Ahmad and Tagliavini, Giuseppe and Rossi, Davide and Benini, Luca},
>   booktitle={2023 Design, Automation & Test in Europe Conference & Exhibition (DATE)},
>   title={TransLib: A Library to Explore Transprecision Floating-Point Arithmetic on Multi-Core IoT End-Nodes},
>   year={2023},
>   volume={},
>   number={},
>   pages={1-2},
>   doi={10.23919/DATE56975.2023.10136916}}
> ```

## Get Started
Each kernel design includes a Python model and a C program. The Python model generates the input dataset, computes the kernel output as a golden reference, and assesses the accuracy using a customizable error metric. Each folder contains a specific test with the golden model generator and a brief description of how to run the test.  
### Prerequisites 
#### Python
These golden models are built on top of PyTorch data types. The following packages need to be installed:
~~~~~shell
pip install torch 
~~~~~

### Float8|e5m2 
to support float8 use https://github.com/ahmad-mirsalari/PyTorch_E5M2 

Navigate to the root directory of the modified PyTorch codebase in the terminal or command prompt.
Run the following command to install PyTorch in editable mode:
~~~~~shell
pip install -e .
~~~~~
The . at the end indicates that the current directory should be installed in editable mode.

Once the installation is complete, you can import the modified version of PyTorch in your Python code just like you would with the regular PyTorch library:
~~~~~shell
import torch
~~~~~
This will import the modified version of PyTorch that you installed in editable mode.
#### PULP-SDK
These tests requires the [PULP-SDK](https://github.com/pulp-platform/pulp-sdk). Once you cloned the PULP-SDK repository and you have the [RISC-V GNU Compiler Toolchain](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain) installed, you need to compile [GVSOC](https://github.com/pulp-platform/pulp-sdk#gvsoc). **Please refer to the links to correctly setup your working environment.**

Here is my suggestion:

1-  First install and compile the [RISC-V GNU Compiler Toolchain](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain#risc-v-gnu-compiler-toolchain).

Follow the next steps in the [RISC-V GNU Compiler Toolchain](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain#risc-v-gnu-compiler-toolchain) repository.

- [Getting the sources](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain#getting-the-sources)
- [Prerequisites](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain#prerequisites)
- [Installation (Pulp)](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain#installation-pulp)

2- Install and compile [PULP-SDK](https://github.com/pulp-platform/pulp-sdk#pulp-sdk).

Please follow the next setups in the [PULP-SDK](https://github.com/pulp-platform/pulp-sdk#pulp-sdk) repository
- [Getting started](https://github.com/pulp-platform/pulp-sdk#getting-started)
- [GVSoC](https://github.com/pulp-platform/pulp-sdk#gvsoc)

3- Finally, test the installation according to [Test execution](https://github.com/pulp-platform/pulp-sdk#test-execution)


**Don't forget to source the file corresponding to the desired configuration when you want to use the TransLib again** :

~~~~~shell
cd pulp-sdk
source configs/pulp-open.sh
~~~~~
## Repository Organization
Each folder contains a specific test with the golden model generator and a brief description of how to run the test.  

Since some kernels don't support mixed-precision in the current version, we have divided the kernels into [fixed-precision test ](./fixed_precision/) and [mixed-precision test](./mixed_precision/).
These are the developed tests:

- [Matrix Multiplication test](./mixed_precision/matmul/) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [Convolution test](./mixed_precision/convolutioncl/) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [DWT test](./fixed_precision/dwt) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [FFT test](./fixed_precision/fft-memsave) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [FIR filter test](./mixed_precision/fir) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [K-means algorithm test](./fixed_precision/kmeans) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point
- [SVM classification test](./mixed_precision/SVM/) for FP32, FP16 and FP16ALT with also vectorial format for half-precision floating-point


## Roadmap

- Extend the support to additional FP data types (e.g., different flavors of 8-bit FP types) 

## License 
 TransLib is released under Apache 2.0, see the [LICENSE](./LICENSE.md) file in the root of this repository for details.

## Acknowledgements
This work was supported by the [APROPOS](https://projects.tuni.fi/apropos/) project (g.a. no. 956090), founded by the European Union’s Horizon 2020 research and innovation program. 


## Contributors
- [Seyed Ahmad Mirsalari](https://github.com/ahmad-mirsalari), University of Bologna,[E-mail](mailto:seyedahmad.mirsalar2@unibo.it)


## 🚀 Contact Me
- [Email](mailto:seyedahmad.mirsalar2@unibo.it)
- [LinkedIn](https://www.linkedin.com/in/ahmad-mirsalari/)
- [Twitter](https://twitter.com/ahmad_mirsalari)
- [APROPOS](https://projects.tuni.fi/apropos/news/pr_esr_3/)


