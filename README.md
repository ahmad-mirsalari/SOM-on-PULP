# Optimizing Self-Organizing Maps for Bacterial Genome Identification on Parallel Ultra-Low-Power Platforms


## Get Started
The kernel design includes a Python model and a C program. The Python model generates the input dataset, computes the kernel output as a golden reference, and assesses the accuracy using a customizable error metric. 
### Prerequisites 
#### Python
These golden models are built on top of PyTorch data types. The following packages need to be installed:
~~~~~shell
pip install torch 
~~~~~

### Float8|e5m2 
to support float8 use [our e5m2 implementation](https://github.com/ahmad-mirsalari/PyTorch_E5M2)

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


**Don't forget to source the file corresponding to the desired configuration when you want to use the the project again** :

~~~~~shell
cd pulp-sdk
source configs/pulp-open.sh
~~~~~
# How to run a test
After the platform and the SDK setup, you can run the test.

## Generating the golden model
If you want to generate a golden model, you can use the [data_generator.py](./data_generator.py) script with the following command:

~~~~~shell
./data_generator.py --I=Train_sequence --T=Test_sequence --R=Number_of_Bacteria --S=slice_length --N=Neurons --float_type= --MAC_flag=false --vec_flag=false
~~~~~

- specifies the floating-point format for data; by default, it is set to `FP32`, but you can also choose `FP16`, `FP16ALT`, and `FP8` formats. **Also, you can run the mixed-precision golden model by using `--float_type=FP_INP,FP_Weight,FP_OUT` (input, SOM weights, output).**
- `MAC_flag` is used to emulate the multiply-and-add operator available on most DSP instruction sets for embedded devices. It can be true or false. To emulate `FP16`,  `FP8`, and `FP16ALT` behavior on PULP, true this flag.
- vector flag to emulate SIMD vector instructions. It can be true or false. To emulate vectorized `FP16`,  `FP8` and `FP16ALT` behavior on PULP, true this flag.
- `I` is the number of train data(e.g., 40000)
- `T` is the number of test data(e.g., 1000)
- `R` is the number of Bacteria(SOM)(e.g., 2).
- `S` is the number of slice lengths (e.g., 8)
- `N` is the number of neurons per each network(e.g., 40000)
The script will generate floating-point data and a reference output of format `fmt` (FP32/FP16/FP8/FP16ALT):

## Running the C code
~~~~~shell
make clean all run stats=1 check=1 w_block=128 i_block=32 vec=1 cores=1 fmt=FP16 verbose=1  IN_ORDER=1
~~~~~
 make clean all run stats=1 check=1 w_block=128 i_block=32  cores=1 fmt=FP16 verbose=1  IN_ORDER=1 
There are several flags useful to activate some functionalities:

- `cores=N_CORES` sets the number of cores used for the execution to `N_CORES`, by default `cores=1`. There is also the ability to run on the Fabric controller by using `FABRIC=1` instead of `cores=N_CORE`.
- `fmt=FP_FMT` specifies the floating-point format for data, by default, it is set to `FP32` but you can also choose `FP16`, `FP8` or `FP16ALT` formats.
- `vec=1` activates vectorial format **only for half-precision floating point (FP16 and FP16ALT) or FP8**
- `check=1` activates the result check
- `verbose=1` prints the wrong results
- `stats=1` activates performance measurement
- `PRINT_RESULTS=1` print outputs of C code
- `w_block` the tile size of the SOM network (The number of neurons must be divisible by this number)
- `i_block` the tile size of the input (The number of input must be divisible by this number)
- `IN_ORDER=1` if you want to use the Vertical Mapping approach. **Please consider that the number of cores should be >1 in the Horizontal mapping mode** 

 

## Roadmap

- Extend the support to additional FP data types (e.g., different flavors of 8-bit FP types) 

## License 
 This project is released under Apache 2.0, see the [LICENSE](./LICENSE.md) file in the root of this repository for details.

## Acknowledgements
This work was supported by the [APROPOS](https://projects.tuni.fi/apropos/) project (g.a. no. 956090), founded by the European Union’s Horizon 2020 research and innovation program. 


## Contributors
- [Seyed Ahmad Mirsalari](https://github.com/ahmad-mirsalari), University of Bologna,[E-mail](mailto:seyedahmad.mirsalar2@unibo.it)


## 🚀 Contact Me
- [Email](mailto:seyedahmad.mirsalar2@unibo.it)
- [LinkedIn](https://www.linkedin.com/in/ahmad-mirsalari/)
- [Twitter](https://twitter.com/ahmad_mirsalari)
- [APROPOS](https://projects.tuni.fi/apropos/news/pr_esr_3/)


