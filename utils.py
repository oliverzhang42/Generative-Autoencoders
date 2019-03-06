import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [3.0, 2.0]
plt.rcParams["axes.grid"] = False

import ot


def display(file_path):
	images = np.load(file_path)
	indices = np.random.choice(range(len(images)), size=16)
	display_img(images[indices])

def display_img(x, single=False, labels=None, columns=4):
    if single:
        img = x.reshape(28, 28)
        plt.figure()
        if labels:
            plt.title(labels)
        plt.imshow(img, cmap='gray')
    else:
        f, subs = plt.subplots(columns, len(x)//columns, sharex='col', sharey='row')
            
        for i in range(len(x)):
            img = x[i].reshape(28, 28)
            if labels:
                plt.title(labels[i])
            subs[i//columns][i%columns].imshow(img, cmap='gray')
    
    plt.show()

def ot_compute_answers(encodings, batch_size):
    inputs = np.random.random(size=encodings.shape)
    answers = []

    for i in range(len(encodings) // batch_size):
        if i % 10 == 0:
              print("Batch Number: {}".format(i))

        real_vec = encodings[i*batch_size: (i+1)*batch_size]
        fake_vec = inputs[i*batch_size: (i+1)*batch_size]
          
        a = np.ones((batch_size, ))
        b = np.ones((batch_size, ))
          
        M = []
          
        for j in range(batch_size):
            costs = []
            
            for k in range(batch_size):
                costs.append(np.sum(np.abs(fake_vec[j] - real_vec[k]))) # TODO: Debug the abs & sum & encodings vs real_vec?
            M.append(costs)
            
        M = np.array(M)
        mapping = ot.emd(a, b, M)
          
        for j in range(batch_size):
            index = np.argmax(mapping[j])
            answers.append(real_vec[index])

    answers = np.array(answers)
    return (inputs, answers)