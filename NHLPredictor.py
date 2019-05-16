import torch
import torch.nn as nn
import xlrd
import numpy as np

#x = torch.tensor(([2, 9], [1, 5], [3, 6], [8, 1], [5, 6], [1, 4], [7, 7], [0, 4], [1, 3]), dtype=torch.float) # 9 X 2 tensor
#y = torch.tensor(([75, 1], [69, 0], [84, 1], [64, 0], [94, 1], [64, 0], [94, 1], [53, 0], [57, 0]), dtype=torch.float) # 9 X 2 tensor
#xPredicted = torch.tensor(([4, 8]), dtype=torch.float64) # 1 X 2 tensor
numpyx = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], dtype = np.float) # initialize x numpy array with 42 features that eventually becomes a tensor 
numpyy = np.array([[0,0]], dtype = np.float) # initialize y numpy array with 1 label that eventually becomes a tensor 
numpyxpred = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], dtype = np.float)

#print(x.size())
#print(y.size())
#print(x)
#print(y)

book = xlrd.open_workbook("C:/Users/psfon/Documents/PyTorch/Project/NHLData.xlsx")
print("The number of worksheets is {0}".format(book.nsheets))
print("Worksheet name(s): {0}".format(book.sheet_names()))
sh = book.sheet_by_index(0)
print("{0} {1} {2}".format(sh.name, sh.nrows, sh.ncols))
#print("Cell D30 is {0}".format(sh.cell_value(rowx=29, colx=3)))
for rx in range(sh.nrows): # loading values for each separate data point
    if (rx > 0): #don't do the first row because that's all names
        #print(sh.row(rx))
        cellsx = sh.row_slice(rowx = rx, start_colx = 2, end_colx = 44) #first two columns are for features
        cellsy = sh.row_slice(rowx = rx, start_colx = 44, end_colx = 46) #first second columns are for labels
        xarr = [] #array for x data point
        yarr = [] #array for y data point
        for cell in cellsx:
            xarr.append(cell.value)
        #print(xarr)
        if (rx != sh.nrows-1):
            numpyx = np.append(numpyx, [xarr], axis = 0) #add data point to x values (training)
        else:
            numpyxpred = np.append(numpyxpred, [xarr], axis = 0) #for testing data points
        for cell in cellsy:
            yarr.append(cell.value)
        if (rx != sh.nrows-1):
            numpyy = np.append(numpyy, [yarr], axis = 0) #add data point to x values

numpyx = np.delete(numpyx,0, axis = 0) # delete initializing with dimensions part of numpy array 
numpyy = np.delete(numpyy,0, axis = 0) # delete initializing with dimensions part of numpy array 
numpyxpred = np.delete(numpyxpred,0, axis = 0) # delete initializing with dimensions part of numpy array 
x = torch.from_numpy(numpyx)
y = torch.from_numpy(numpyy)
xPredicted = torch.from_numpy(numpyxpred)

# scale units
x_max, _ = torch.max(x, 0)
#xPredicted_max, _ = torch.max(xPredicted, 0)

x = torch.div(x, x_max)
xPredicted = torch.div(xPredicted, x_max)
#print(x)
#print(y)
#print(xPredicted)
print(x.size())

class Neural_Network(nn.Module):
    def __init__(self, ):
        super(Neural_Network, self).__init__()
        # parameters
        # TODO: parameters can be parameterized instead of declaring them here
        self.inputSize = 42
        self.hiddenSize1 = 3
        self.hiddenSize2 = 3
        self.hiddenSize3 = 3
        self.outputSize = 2
        
        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize1, dtype=torch.float64) # 3 X 42 tensor
        self.W2 = torch.randn(self.hiddenSize1, self.hiddenSize2, dtype=torch.float64) # 3 X 3 tensor
        self.W3 = torch.randn(self.hiddenSize2, self.hiddenSize3, dtype=torch.float64) # 3 X 3 tensor
        self.W4 = torch.randn(self.hiddenSize3, self.outputSize, dtype=torch.float64) # 3 X 2 tensor
        
    def forward(self, x):
        self.z = torch.matmul(x, self.W1) # 3 X 3 ".dot"
        self.z2 = self.sigmoid(self.z) # activation function input
        self.z3 = torch.matmul(self.z2, self.W2)
        self.z4 = self.sigmoid(self.z3) # activation function first layer
        self.z5 = torch.matmul(self.z4, self.W3)
        self.z6 = self.sigmoid(self.z5) # activation function second layer
        self.z7 = torch.matmul(self.z6, self.W4)
        o = self.sigmoid(self.z7) # final activation function
        return o
        
    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))
    
    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)
    
    def backward(self, x, y, o):
        self.o_error = y - o # error in output
        self.o_delta = self.o_error * self.sigmoidPrime(o) # derivative of sig to error
        self.z6_error = torch.matmul(self.o_delta, torch.t(self.W4))
        self.z6_delta = self.z6_error * self.sigmoidPrime(self.z6)
        self.z4_error = torch.matmul(self.z6_delta, torch.t(self.W3))
        self.z4_delta = self.z4_error * self.sigmoidPrime(self.z4)
        self.z2_error = torch.matmul(self.z4_delta, torch.t(self.W2))
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
        self.W1 += torch.matmul(torch.t(x), self.z2_delta)
        self.W2 += torch.matmul(torch.t(self.z2), self.z4_delta)
        self.W3 += torch.matmul(torch.t(self.z4), self.z4_delta)
        self.W4 += torch.matmul(torch.t(self.z6), self.o_delta)
        
    def train(self, x, y):
        # forward + backward pass for training
        o = self.forward(x)
        self.backward(x, y, o)
        
    def saveWeights(self, model):
        # we will use the PyTorch internal storage functions
        torch.save(model, "NN")
        # you can reload model with all the weights and so forth with:
        # torch.load("NN")
        
    def predict(self, test):
        print ("Predicted data based on trained weights: ")
        print ("Input (scaled): \n" + str(test))
        print ("Output: \n" + str(self.forward(test)))


NN = Neural_Network()
for i in range(500):  # of training steps
    print ("#" + str(i) + " Loss: " + str(torch.mean((y - NN(x))**2).detach().item()))  # mean sum squared loss
    NN.train(x, y)
NN.saveWeights(NN)
NN.predict(xPredicted)