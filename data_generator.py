import copy
import math
import sys

import numpy as np
from numpy import random
import torch
import scipy.io
import argparse

'''
If you are working offline in local computer, just set folder_addr to '.' or absolute address of project folder in your computer, 
but if you are working in online, you may set folder_addr to the path of the project in the Internet.
'''
folder_addr = "."


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
            temp[i][j] = IN[i][j]
           
            
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
    mat = scipy.io.loadmat(folder_addr + '/dataset/'+'bacteria.' + str(index) + '.short.mat')
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

    if check_all_float(dt): # if all data types are float32 we can use the python default functions
        distSq = (torch.abs(SOM - x)).sum(dim=1)
        return torch.argmin(distSq, dim=None)
    else:
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
                    temp[i] += diff
            return torch.argmin(temp, dim=None)
        else: #Vectorial mode
            if dt[0] == torch.float8:
              vec_step = 4
            else:
              vec_step = 2
            temp = torch.zeros(SOM.shape[0], dtype=dt[2])
            temp1 = torch.zeros(vec_step , dtype=dt[2])
            for i in range(SOM.shape[0]):
              for k in range(vec_step):
                temp1[k] = 0
              for j in range(0,SOM.shape[1],vec_step):
                for k in range(vec_step):
                  temp1[k] += (torch.abs(SOM[i, j+k] - x[j+k]))
                
              for k in range(vec_step):
                temp[i] += temp1[k] 
            return torch.argmin(temp, dim=None)


# Update the weights of the SOM cells when given a single training example
# and the model parameters along with BMU coordinates as a tuple
def update_weights(SOM, train_ex, beta, g, dt, mac_flag):
    # Change all cells in a neighborhood of BMU
    N = SOM.shape[0]
    M = SOM.shape[1]
    for i in range(N):

        dist = (N / 2) - torch.abs(torch.abs(i - g) - (N / 2))
        power = (torch.pow(2, dist))
        power_temp = (beta / power)
        if ((dt[1] == torch.float16 and dist > 14 ) or (dt[1]==torch.bfloat16) and dist > 19) or ((dt[1] == torch.float32) and dist > 19) or ((dt[1] == torch.float8) and dist > 14):#if abs(power_temp) < torch.finfo(dt).eps or dist > 15:  # if abs(power) > (2 ** 31 - 1):  # abs(power) > (1 << 31) - 1)
            # https://stackoverflow.com/questions/45528637/checking-integer-overflow-in-python
            # print("underflow!!")
            continue
        for j in range(M):
            som_ij = torch.tensor(0, dtype= dt[2])
            x_j = torch.tensor(0, dtype= dt[2])
            som_ij = SOM[i,j].type(dt[2])
            x_j = train_ex[j].type(dt[2])
            a = (som_ij - x_j)
            temp = som_ij
            if mac_flag == 'true':
                a = a.type(torch.float32)
                power_temp = power_temp.type(torch.float32)
                temp = temp.type(torch.float32)
            temp -= power_temp * a
            if mac_flag == 'true':
                temp = temp.type(dt[1])
            SOM[i, j] = temp

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
            g = find_BMU(SOM, train_ex, dt, vec_flag)
            SOM = update_weights(SOM, train_ex, beta, g, dt, mac_flag)
            # Update beta
            beta = max(beta * decay_factor, beta_min)
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
         
      
    C_false = sum(x != y for x, y in zip(test_DNA, np.argmin(score, axis=1)))
    print("Number of differences:", C_false)
    print("Classification error:", (C_false / test_itr) * 100)
    differences = [(i, x, y) for i, (x, y) in enumerate(zip(test_DNA, np.argmin(score, axis=1))) if x != y]


def inference_SOM(SOM, test_data, dt):
    temp = torch.tensor(0, dtype=dt[2])
    error = torch.tensor(0, dtype=dt[2])
    distSq = torch.tensor(0, dtype=dt[2])

    for test_ex in test_data:
        SOM = SOM.type(dt[2])
        test_ex = test_ex.type(dt[2])
        distSq = (torch.abs(SOM - test_ex)).sum(dim=1)
        error = distSq.min()
        temp += error
    return temp




def merge_mats(L_mats, S_mat, N, M, idx):
    for i in range(N):
        for j in range(M):
            L_mats[(idx * N * M) + (i * M) + j] = S_mat[i, j]
    return L_mats

def split_soms(SOMs, N, M, idx):
    SOM = torch.zeros(N,M)
    for i in range(N):
        for j in range(M):
            SOM[i, j] = SOMs[(idx * N * M) + (i * M) + j]

    return SOM

def quantization_error(true, pred):
    all_diff = torch.divide(torch.abs(true - pred),true)
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

        # load data in fp32
        train = read_data(i, size_data_train, slice_length, enc_bits, torch.float32) #

        # convert input data to dt[0]
        train_conv = matrix_init(train, data_types[0])

        # merge all input data in dt[0] data type for the C code
        trains_data = merge_mats(trains_data, train_conv, I, M, i)

        # Initialize the SOM randomly in fp32
        SOM_init = torch.randn((N, M), dtype=torch.float32)

        # convert SOM init to dt[1]
        SOM_init_conv = matrix_init(SOM_init, data_types[1])
        # SOM_ref = copy.deepcopy(SOM_init_ref)

        # save all SOMs init for the C code
        SOMs_init_conv = merge_mats(SOMs_init_conv, copy.deepcopy(SOM_init_conv), N, M, i)
        # Memory estimations

        # Train the ref SOM in fp32
        dt_fp32 = [torch.float32,torch.float32,torch.float32]
        SOM_ref = train_SOM(SOM_init, train,  dt_fp32, epochs=epoch, mac_flag='false', vec_flag='false', print_flag='false')

        if not check_all_float(data_types): # there is at least one non float32 data type
            # Train the conv SOM
            print(f"Training of the SOM {i} in {data_types} is started")

            SOM_res = train_SOM(SOM_init_conv, train_conv, data_types, epochs=epoch, mac_flag=mac_flag,vec_flag=vec_flag, print_flag='true')
            SOMs_res = merge_mats(SOMs_res, SOM_res, N, M, i)
        else:#all data types are float32 and we can use the ref results
            SOM_res = SOM_ref
            SOMs_res = merge_mats(SOMs_res, SOM_ref, N, M, i)

        print("MSE is ", mean_squared_error(SOM_ref, SOM_res).item())
        print(f"quantization_error is {quantization_error(SOM_ref, SOM_res).item()}")
        test_data = read_data(test_DNA , size_data_test, slice_length, enc_bits, data_types[0])


        # Calculate the score (SOM inference)
        
        score[i] = inference_SOM(SOM_res, test_data, data_types)

    save_data_into_hfile(R, N, M, I, epoch, trains_data, SOMs_res, SOMs_init_conv, test_data, score[0])
    print("The target(true) DNA is %d \nThe winner SOM is %d " % (test_DNA, np.argmin(score) ))

    
    test (N, M, R, slice_length, enc_bits,size_data_test, SOMs_res,data_types )
    
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    task_dna()
