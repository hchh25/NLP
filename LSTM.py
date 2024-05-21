import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


w_f_xh = np.array([1.63])
w_f_sh = np.array([2.7])
b_f = np.array([1.62])

w_i_LR_xh = np.array([1.65])
w_i_LR_sh = np.array([2.0])
b_i_LR = np.array([0.62])

w_i_UPD_xh = np.array([0.94])
w_i_UPD_sh = np.array([1.41])
b_i_UPD = np.array([-0.32])

w_o_xh = np.array([-0.19])
w_o_sh = np.array([4.38])
b_o = np.array([0.59])



def LSTM(x, short_term_mem, long_term_mem):
	# Forget Gate
	f = sigmoid(np.dot(x, w_f_xh) + np.dot(short_term_mem, w_f_sh) + b_f)

	# modify long-term memory:
	long_term_mem = np.multiply(f, long_term_mem)
	
	# Input Gate
	i_LR = sigmoid(np.dot(x, w_i_LR_xh) + np.dot(short_term_mem, w_i_LR_sh) + b_i_LR)

	i_UPD = np.tanh(np.dot(x, w_i_UPD_xh) + np.dot(short_term_mem, w_i_UPD_sh) + b_i_UPD)
	
	# modify long-term memory:
	long_term_mem = np.add(np.multiply(i_LR, i_UPD), long_term_mem)
	
	# Output Gate
	o = sigmoid(np.dot(x, w_o_xh) + np.dot(short_term_mem, w_o_sh) + b_o)
	
	# modify short-term memory:
	short_term_mem = np.multiply(o, np.tanh(long_term_mem))
	
	return short_term_mem, long_term_mem

# print(LSTM(1,1,2))

x_1=[0, 0.5, 0.25, 1]
x_2=[1, 0.5, 0.25, 1]

s1 = 0
l1 = 0
s2 = 0
l2 = 0


for i in range(len(x_1)):
	s1, l1 = LSTM(x_1[i], s1, l1)
	s2, l2 = LSTM(x_2[i], s2, l2)
	print("s1: ", s1, "l1: ", l1, "s2: ", s2, "l2: ", l2)








# import matplotlib.pyplot as plt
# # two plots
# fig, axs = plt.subplots(2)
# fig.suptitle('LSTM - Sequential Input')
# axs[0].plot(x_1[:4])
# axs[0].scatter(4, 0, color='red')
# axs[0].plot([3,4], [1,0], color='red', ls='--')
# axs[0].set_title('Stock 1')
# axs[1].plot(x_2[:4])
# axs[1].scatter(4, 1, color='red')
# axs[1].plot([3,4], [1,1], color='red', ls='--')
# axs[1].set_title('Stock 2')
# plt.tight_layout()
# plt.show()
