import copy
import math
import sys

import numpy as np
from numpy import random
import torch
import scipy.io
import argparse



def save_data_into_hfile(R, N, M, I, epoch, inp, ref, som, test, score):
    # Generate header file
    g = open('param.h', 'w')
    g.write('\
#define R_g %s \n\
#define M_g %s \n\
#define N_g %s\n\n' % (R, M, N))
    g.close()
    f = open('data.h', 'w')
    f.write('\
#define R %s\n\
#define N %s\n\
#define M %s\n\
#define Epoch %s\n\
#define K %s\n\
#define score_ref %s\n\
#define I %s\n\n' % (R, N, M, epoch, test.shape[0], score.item(), I))
    write_matrix(inp, 'input_train', f, R, inp.dtype)
    # write_matrix(test, 'input_test', f, inp.dtype)

    write_matrix(ref, 'ref', f, R, ref.dtype)
    write_matrix(som, 'SOM_N', f, R, som.dtype)
    f.close()


def write_matrix(matrix_to_write, name, file_pointer, R, float_type):
    rem_part = ''
    matrix_string = ''
    if 'ref' in name:
        file_pointer.write("PI_L2 Pixel %s[R * N * M] = {" % name)

    elif 'input_train' in name:
        file_pointer.write("PI_L2 Pixel %s[R * I * M] = {" % name)
    elif 'input_test' in name:
        file_pointer.write("PI_L2 Pixel %s[K * M] = {" % name)
    else:
        file_pointer.write("PI_L2 Pixel %s[R * N * M] = {" % name)

    #print(f'data type is {float_type}')
    if float_type == torch.float32:
        rem_part = ")"
    elif float_type == torch.float16:
        rem_part = ", dtype=torch.float16)"
    elif float_type == torch.bfloat16:
        rem_part = ", dtype=torch.bfloat16)"
    elif float_type == torch.float8:
        rem_part = ", dtype=torch.float8)"

    sz0 = matrix_to_write.shape[0]
    for i in range(sz0):
        matrix_string += str(matrix_to_write[i].item()).replace('tensor(', '').replace(rem_part, '')
        matrix_string += ','
    file_pointer.write("%s" % matrix_string)
    file_pointer.write("};\n")


def find_dividable(a, b):
    for i in range(b, 0, -1):
        if (a % i) == 0:
            return i


def memory_estimation(N, M, I, S, data_type):
    # N : Neurons
    # M : Features
    # I : Input_Size
    # S : Number of SOMs
    Kb = 1024
    L1 = 56 * Kb  # L1 in Bytes
    # Define number of bytes per float
    if data_type == torch.float32:
        data_size = 4
    elif data_type == torch.float16 or data_type == torch.bfloat16:
        data_size = 2

    Weights = N * M * data_size  # Memory  for only one SOM(Weights) in bytes

    # calculate the memory for S SOMs
    SOMs_b = S * Weights
    SOMs_kb = SOMs_b / 1024
    print("Weight Mem for %s SOM is: %s Bytes(%s KB)" % (S, SOMs_b, SOMs_kb))
    Inputs = I * M * data_size * S  # Memory for only input in Bytes
    Inputs_kb = SOMs_b / 1024
    print("Mem for %s Input is: %s Bytes(%s KB)" % (S, Inputs, Inputs_kb))
    for i in range(3, int(math.log2(N)) + 1):
        print("---------------------- Config Number %s -------------------" % i)
        w_number = int(pow(2, i))  # number of weights for each block (only one SOM)
        w_memory = w_number * M * data_size  # weight memory for one block(only one SOM)
        w_memory_soms = w_memory * S  # weight memory for s Soms (S SOMs)
        w_memory_soms_db = w_memory_soms * 2  # enable double buffering

        print(" ----------------- Weights--------------------")
        print(
            "L1 block for double buffering: %s -> dividing by 2(double buffering): %s -> each SOM: %s -> Number: %s" % (
                w_memory_soms_db, w_memory_soms, w_memory, w_number))
        Avb_L1 = L1 - w_memory_soms_db  # Available L1 in Bytes

        # calculate the number of input for each SOM
        N_input_all = int(Avb_L1 / (M * data_size))  # all soms
        N_input_db = int(N_input_all / (S))  # for each SOM: we have S input dataset with M features,
        # enabling double buffering, 2 is for double buffering
        N_input_som = int(N_input_db / 2)  # we must copy N_input from each dataset with double buffering

        z = find_dividable(I, N_input_som)
        print(
            "Available L1: %s -> it is enough for %s inputs -> we have %s dataset: %s -> we use double buffering: %s, "
            "it should be dividable: %s " % (
                Avb_L1, N_input_all, S, N_input_db, N_input_som, z))

    return z, w_number


def relative_absolute_error(true, pred):
    true_mean = torch.mean(true)
    squared_error_num = torch.sum(torch.abs(true - pred))
    squared_error_den = torch.sum(torch.abs(true - true_mean))
    rae_loss = squared_error_num / squared_error_den
    return rae_loss


def mean_squared_error(true, pred):
    squared_error = torch.square(true - pred)
    sum_squared_error = torch.sum(squared_error)
    size = true.size(dim=0) * true.size(dim=1)
    mse_loss = sum_squared_error / size
    return mse_loss


def matrix_init(IN, dt):
    temp = torch.zeros((IN.shape[0], IN.shape[1]), dtype=dt)
    # iterate through rows of IN
    for i in range(IN.shape[0]):
        # iterate through columns of IN
        for j in range(IN.shape[1]):
            #print("init value is",IN[i][j])
            temp[i][j] = IN[i][j]
            #print("convet value is",temp[i][j])
            
    return temp


def select_dtypes(user_dtypes, num_param):
    types_dict = {
        "FP8": torch.float8,
        "FP16": torch.float16,
        "FP16ALT": torch.bfloat16,
        "FP32": torch.float32
    }
    dtypes = []
    if len(user_dtypes) == 1:
        for i in range(num_param):
            dtypes.append(types_dict[user_dtypes[0]])
    elif len(user_dtypes) == num_param:
        for i in range(num_param):
            dtypes.append(types_dict[user_dtypes[i]])
    else:
        for i in range(len(user_dtypes)):
            dtypes.append(types_dict[user_dtypes[i]])
        if 'FP32' in user_dtypes:
            for i in range(len(user_dtypes), num_param):
                dtypes.append(types_dict["FP32"])
        elif 'FP16' in user_dtypes:
            for i in range(len(user_dtypes), num_param):
                dtypes.append(types_dict["FP16"])
        elif 'FP8' in user_dtypes:
            for i in range(len(user_dtypes), num_param):
                dtypes.append(types_dict["FP8"])
        else:
            for i in range(len(user_dtypes), num_param):
                dtypes.append(types_dict["FP16ALT"])
    return dtypes


def get_inital_config():
    # get arguments  and data format
    parser = argparse.ArgumentParser()
    parser.add_argument('--I', default= 1024) # train size
    parser.add_argument('--T', default= 10) # test size
    parser.add_argument('--R', default= 2) # number of bacteria's(SOMs)
    parser.add_argument('--S', default= 8) # slice length
    parser.add_argument('--N', default= 128) # number of neurons

    parser.add_argument('--MAC_flag', default="false")
    parser.add_argument('--vec_flag', default="false")
    parser.add_argument('--float_type', default='FP32')
    args = parser.parse_args()

    I = int(args.I)
    T = int(args.T)
    R = int(args.R)
    S = int(args.S)
    N = int(args.N)
    
    mac_flag = str(args.MAC_flag)
    vec_flag = str(args.vec_flag)
    bits = args.float_type.split(",")
    return I,T,R,S,N,bits, mac_flag, vec_flag


def encode(input_seq, enc_bits, dt):
    out_seq = torch.zeros(enc_bits * len(input_seq), dtype=dt)
    out_seq.requires_grad_(False)
    length = len(input_seq)

    if enc_bits == 2:
        for i in range(length):
            if input_seq[i] == 'A':
                out_seq[2 * i] = -1
                out_seq[2 * i + 1] = 0
            elif input_seq[i] == 'T':
                out_seq[2 * i] = 1
                out_seq[2 * i + 1] = 0
            elif input_seq[i] == 'C':
                out_seq[2 * i] = 0
                out_seq[2 * i + 1] = -1
            elif input_seq[i] == 'G':
                out_seq[2 * i] = 0
                out_seq[2 * i + 1] = 1
            else:
                out_seq[2 * i] = 0
                out_seq[2 * i + 1] = 0
    else:
        for i in range(length):
            if input_seq[i] == 'A':
                out_seq[1 * i] = 0
                # out_seq[2 * i + 1] = 0
            elif input_seq[i] == 'T':
                out_seq[1 * i] = -1
                # out_seq[2 * i + 1] = 0
            elif input_seq[i] == 'C':
                out_seq[1 * i] = 2
                # out_seq[2 * i + 1] = -1
            elif input_seq[i] == 'G':
                out_seq[1 * i] = -2
                # out_seq[2 * i + 1] = 1
            else:
                out_seq[1 * i] = 3
                # out_seq[2 * i + 1] = 0
    return out_seq


def random_slice(Input, Input_length, slice_length):
    # np.random.seed(0)
    # way 1
    # start = list(range(0, Input_length - slice_length))
    # random.shuffle(start)
    # print(start)

    # way 2
    start = np.random.randint(low=0, high=Input_length - slice_length)
    seq = Input[start:start + slice_length]

    return seq


def read_data(index, size_data, slice_length, enc_bits, dt):
    mat = scipy.io.loadmat('bacteria.' + str(index) + '.short.mat')
    mat = str(mat['var_data_short'])
    arr = mat[mat.find("['") + 2:mat.find("']")]
    train = list(arr)

    data = torch.zeros((size_data, enc_bits * slice_length), dtype=dt)
    data.requires_grad_(False)
    for i in range(size_data):
        temp = random_slice(train, len(train), slice_length)
        data[i,] = encode(temp, enc_bits, dt)

    return data


def check_all_float(datatypes):#if all data types are float32 
    result = len(set(datatypes)) == 1  
    if result : #All Elements in List are Equal = fixed precision
        if torch.float32 in datatypes:#All data types are float32
            return True
        else:
            return False
    else: #All Elements in List are Not Equal = Transprecision
        return False



def find_BMU(SOM, x, dt, vec_flag='false'):
    # print("input", x)
    # print("SOM", SOM)

    if check_all_float(dt): # if all data types are float32 we can use the python default functions
        distSq = (torch.abs(SOM - x)).sum(dim=1)
        # print("distSq is %s" %(distSq))
        return torch.argmin(distSq, dim=None)
    else:
        # print("I am here ")
        if vec_flag == 'false':
            temp = torch.zeros(SOM.shape[0], dtype= dt[2])
            diff = torch.zeros(SOM.shape[0], dtype= dt[2])
            som_ij = torch.tensor(0, dtype= dt[2])
            x_j = torch.tensor(0, dtype= dt[2])
            for i in range(SOM.shape[0]):
                for j in range(SOM.shape[1]):

                    som_ij = SOM[i,j].type(dt[2])
                    x_j = x[j].type(dt[2])
                    diff = (torch.abs(som_ij - x_j))
                    #print(f"som  is {som_ij.item()} x  is {x_j.item()}, diff is {diff.item()}")
                    #diff = diff.type(torch.float32)
                    temp[i] += diff
                    #print("temp is %s i is %s" %(temp[i].item(), i))
            return torch.argmin(temp, dim=None)
        else:
            #print("I am here sfsdfs")
            if (dt[0] == torch.float8):
                temp = torch.zeros(SOM.shape[0], dtype=dt[2])
                for i in range(SOM.shape[0]):
                    temp1 = torch.zeros(1, dtype=dt[2])
                    temp2 = torch.zeros(1, dtype=dt[2])
                    temp3 = torch.zeros(1, dtype=dt[2])
                    temp4 = torch.zeros(1, dtype=dt[2])
                    for j in range(0,SOM.shape[1],4):
                        temp1 += (torch.abs(SOM[i, j] - x[j]))
                        temp2 += (torch.abs(SOM[i, j+1] - x[j+1]))
                        temp3 += (torch.abs(SOM[i, j+2] - x[j+2]))
                        temp4 += (torch.abs(SOM[i, j+3] - x[j+3]))
                        #temp[i] += (torch.abs(SOM[i, j] - x[j]))
                    temp[i] = temp1 + temp2 + temp3 + temp4
                    #print("temp is %s i is %s" %(temp[i].item(), i))
                return torch.argmin(temp, dim=None)
            else:
                temp = torch.zeros(SOM.shape[0], dtype=dt[2])
                for i in range(SOM.shape[0]):
                    temp1 = torch.zeros(1, dtype=dt[2])
                    temp2 = torch.zeros(1, dtype=dt[2])
                    for j in range(0,SOM.shape[1],2):
                        temp1 += (torch.abs(SOM[i, j] - x[j]))
                        temp2 += (torch.abs(SOM[i, j+1] - x[j+1]))
                        #temp[i] += (torch.abs(SOM[i, j] - x[j]))
                    temp[i] = temp1 + temp2
                    # print("temp is %s i is %s" %(temp[i].item(), i))
                return torch.argmin(temp, dim=None)


# Update the weights of the SOM cells when given a single training example
# and the model parameters along with BMU coordinates as a tuple
def update_weights(SOM, train_ex, beta, g, dt, mac_flag):
    # Change all cells in a neighborhood of BMU
    N = SOM.shape[0]
    M = SOM.shape[1]

    # power_temp = torch.zeros(1, dtype=dt)
    # dist_func = torch.zeros(M, dtype=dt)
    for i in range(N):

        dist = (N / 2) - torch.abs(torch.abs(i - g) - (N / 2))

        # print("i is ", i)
        # print("beta is %s , dist is %s "%(beta, dist))
        power = (torch.pow(2, dist))
        #power = 1 << dist
        #
        # if abs(power) > (2 ** 63 - 1):
        #     # print("overflow for i= %s dis is %s power is %s " %(i, dist, power))
        #     continue
        # print("power", power)
        # print("update for N= is %s dist %s , power %s" % (i, dist, power))
        # print("dist is ", dist.item())
        # power = power.type(dt)
        # print("power", power)
        # beta = beta.type(torch.float32)
        # power = power.type(torch.float32)
        power_temp = (beta / power)
        # print(f"power_temp {power_temp.dtype}")
        #power_temp = power_temp.type(dt[2])#comment this line
        #print(dt)
        if ((dt[1] == torch.float16 and dist > 14 ) or (dt[1]==torch.bfloat16) and dist > 19) or ((dt[1] == torch.float32) and dist > 19) or ((dt[1] == torch.float8) and dist > 14):#if abs(power_temp) < torch.finfo(dt).eps or dist > 15:  # if abs(power) > (2 ** 31 - 1):  # abs(power) > (1 << 31) - 1)
            # https://stackoverflow.com/questions/45528637/checking-integer-overflow-in-python
            # print("underflow!!")
            continue

        #print(f"i is {i}, beta is {beta}, dis is {dist}, power is {power}, powertemp is {power_temp}")
        for j in range(M):
            som_ij = torch.tensor(0, dtype= dt[2])
            x_j = torch.tensor(0, dtype= dt[2])
            som_ij = SOM[i,j].type(dt[2])
            x_j = train_ex[j].type(dt[2])
            a = (som_ij - x_j)
            #a = a.type(dt[2])#comment this line
            temp = som_ij
            if mac_flag == 'true':
                a = a.type(torch.float32)
                power_temp = power_temp.type(torch.float32)
                temp = temp.type(torch.float32)
            temp -= power_temp * a

            #print(f"temp {temp.item()} ")
            # print("mult is ", (power_temp * a).item())
            # print("temp is ", temp.item())
            if mac_flag == 'true':
                temp = temp.type(dt[1])
            #print(f"j is {j},  som is {SOM[i,j].item()} [before updating]")
            SOM[i, j] = temp
            #print(f"j is {j},  som is {SOM[i,j].item()} [after updating]")
            # print(f"som {SOM[i,j].dtype}, temp {temp.dtype} ")
            # print("j is %s, after update som is %s "%( j, SOM[i, j].item()))
        # dist_func = power_temp * (SOM[i, :] - train_ex)
        # SOM[i, :] -= dist_func

    return SOM


# Main routine for training an SOM. It requires an initialized SOM grid
# or a partially trained grid as parameter
def train_SOM(SOM, train_data, dt, epochs=10, mac_flag='false', vec_flag='false',print_flag='false'):
    beta = torch.tensor(1.0, dtype=dt[2])
    decay_factor = torch.tensor(0.99, dtype=dt[2])
    beta_min =  torch.tensor(0.01, dtype=dt[2])
    for _ in np.arange(0, epochs):
        # shuffle the train data based on the matlab code (KTH)
        # rand_index = torch.randperm(len(train_data))
        # train_data = copy.deepcopy(train_data[rand_index])
        for train_ex in train_data:
            #print("x is ", train_ex)
            g = find_BMU(SOM, train_ex, dt, vec_flag)
            # if print_flag == 'true':
            #print("location is ",g.item())
            SOM = update_weights(SOM, train_ex, beta, g, dt, mac_flag)
            # Update beta
            beta = max(beta * decay_factor, beta_min)
            #beta = beta.type(torch.float32)
            #print(f"beta is {beta}")
    return SOM

def test (N, M, R, slice_length, enc_bits,size_data_test, SOMs_res,dt ):
    # # test all soms in dt
    test_itr = 1000
    test_DNA = np.zeros(test_itr)
    score = np.zeros((test_itr, R))
    for i in range (test_itr):
      r= random.randint(0, R)  # set the test dataset from (0 to 9)
      
      SOM = torch.zeros(N * M, dtype=dt[1])
      test_data = read_data(r, size_data_test, slice_length, enc_bits, dt[0])
      test_DNA[i] = r
      
      for j in range(R):
         SOM = split_soms(SOMs_res, N, M, j)
         score[i,j] = inference_SOM(SOM, test_data, dt)
         
      
    #print("Final Scores: \n", score)
    #print("The target(true) DNA is %d \nThe pre SOM is %d " % (test_DNA, np.argmin(score, axis=1)))
    C_false = sum(x != y for x, y in zip(test_DNA, np.argmin(score, axis=1)))
    print("Number of differences:", C_false)
    print("Classification error:", (C_false / test_itr) * 100)
    differences = [(i, x, y) for i, (x, y) in enumerate(zip(test_DNA, np.argmin(score, axis=1))) if x != y]

    '''print("Indices of differences:")
    for index, value1, value2 in differences:
      print(f"Index: {index}, correct DNA: {value1}, predict class: {value2}")'''

def inference_SOM(SOM, test_data, dt):
    temp = torch.tensor(0, dtype=dt[2])
    error = torch.tensor(0, dtype=dt[2])
    # SOM=SOM.type(torch.float32)
    distSq = torch.tensor(0, dtype=dt[2])


    for test_ex in test_data:
        SOM = SOM.type(dt[2])
        test_ex = test_ex.type(dt[2])
        distSq = (torch.abs(SOM - test_ex)).sum(dim=1)
        # print("INPUT", test_ex)
        # print("distSq",distSq)
        # print("distSq", distSq.min())
        error = distSq.min()
        temp += error
        #print("temp", temp, temp.dtype)

    return temp




def merge_mats(L_mats, S_mat, N, M, idx):
    for i in range(N):
        for j in range(M):
            L_mats[(idx * N * M) + (i * M) + j] = S_mat[i, j]
            # print("new data in the lcation of  %s , i %s j %s, val %s "%( (idx * N * M) + (i * M) + j ,i,j, SOM[i,j]))
    return L_mats

def split_soms(SOMs, N, M, idx):
    SOM = torch.zeros(N,M)
    #print(f"start index {(idx * N * M)}")
    for i in range(N):
        for j in range(M):

            SOM[i, j] = SOMs[(idx * N * M) + (i * M) + j]
            # print("new data in the lcation of  %s , i %s j %s, val %s "%( (idx * N * M) + (i * M) + j ,i,j, SOM[i,j]))
    return SOM

def quantization_error(true, pred):
    # print(f" true dim {true.shape}")
    all_diff = torch.divide(torch.abs(true - pred),true)
    # print(f" all_diff dim {all_diff.shape}")
    return (torch.mean(all_diff)) * 100
    
   
def task_dna():
    # torch.manual_seed(0)
    
    size_data_train, size_data_test, R, slice_length, N, bits, mac_flag, vec_flag = get_inital_config() 

    epoch = 1
    I = size_data_train   # data we need to train
    enc_bits = 1
    M = enc_bits * slice_length
    score = np.zeros(R)
    score_1 = np.zeros((R,R))
    # set the data types based on the parser input
    data_types = select_dtypes(bits, 3) # input, weights, output
    test_DNA = random.randint(0, R)  # set the test dataset from (0 to 9)
    SOMs_init_conv = torch.zeros(R * N * M, dtype=data_types[1])
    SOMs_res = torch.zeros(R * N * M, dtype=data_types[1])
    trains_data = torch.zeros(R * I * M, dtype=data_types[0])  # save the dt data for the c code
    test_data = torch.zeros(size_data_test * M, dtype=data_types[0])
    print("Training is started...")
    for i in range(0, R):
        print(f"Training of the SOM {i} is started")

        # Load the i_th train dataset
        # train = pd.read_csv('train_data_DNA' + str(i) + '.csv', header=None)
        # train = np.asarray

        # load data in fp32
        train = read_data(i, size_data_train, slice_length, enc_bits, torch.float32) #

        # convert input data to dt[0]
        train_conv = matrix_init(train, data_types[0])

        # merge all input data in dt[0] data type for the C code
        trains_data = merge_mats(trains_data, train_conv, I, M, i)

        # Initialize the SOM randomly in fp32
        SOM_init = torch.randn((N, M), dtype=torch.float32)
        # print(SOM_init)

        # convert SOM init to dt[1]
        SOM_init_conv = matrix_init(SOM_init, data_types[1])
        # SOM_ref = copy.deepcopy(SOM_init_ref)

        # save all SOMs init for the C code
        SOMs_init_conv = merge_mats(SOMs_init_conv, copy.deepcopy(SOM_init_conv), N, M, i)
        # Memory estimations

        # Train the ref SOM in fp32
        dt_fp32 = [torch.float32,torch.float32,torch.float32]
        SOM_ref = train_SOM(SOM_init, train,  dt_fp32, epochs=epoch, mac_flag='false', vec_flag='false', print_flag='false')
        # print("Training of the SOM %d is finished" % i)

        if not check_all_float(data_types): # there is at least one non float32 data type
            # Train the conv SOM
            print(f"Training of the SOM {i} in {data_types} is started")

            SOM_res = train_SOM(SOM_init_conv, train_conv, data_types, epochs=epoch, mac_flag=mac_flag,vec_flag=vec_flag, print_flag='true')
            # print("Training of the SOM %d is finished" % i)
            # print("SOM res", SOM_res)
            SOMs_res = merge_mats(SOMs_res, SOM_res, N, M, i)
        else:#all data types are float32 and we can use the ref results
            SOM_res = SOM_ref
            SOMs_res = merge_mats(SOMs_res, SOM_ref, N, M, i)

        # Load the test dataset
        # test = pd.read_csv('test_data.csv', header=None)
        # test = np.asarray(test)

        print("MSE is ", mean_squared_error(SOM_ref, SOM_res).item())
        print(f"quantization_error is {quantization_error(SOM_ref, SOM_res).item()}")
        test_data = read_data(test_DNA , size_data_test, slice_length, enc_bits, data_types[0])


        # Calculate the score (SOM inference)
        
        score[i] = inference_SOM(SOM_res, test_data, data_types)

        '''for j in range(R):
            test_data = read_data(j, size_data_test, slice_length, enc_bits, data_types[0])
            # Calculate the score (SOM inference)
            score_1[i,j] = inference_SOM(SOM_res, test_data, data_types)
    print("Final Scores1: \n", score_1)
    print("The winner SOMs are ", np.argmin(score_1, axis=1))'''
    # inp_b, weight_b = memory_estimation(N, M, I, R, dt)
    save_data_into_hfile(R, N, M, I, epoch, trains_data, SOMs_res, SOMs_init_conv, test_data, score[0])
    #print("Final Scores: \n", score)
    print("The target(true) DNA is %d \nThe winner SOM is %d " % (test_DNA, np.argmin(score) ))

    
    test (N, M, R, slice_length, enc_bits,size_data_test, SOMs_res,data_types )
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    task_dna()