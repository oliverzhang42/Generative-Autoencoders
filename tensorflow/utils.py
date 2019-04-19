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

def ot_compute_answers(inputs, encodings, batch_size, verbose=True, ratio=1):
    # TODO: Find some way of getting rid of ratio. Currently it doesn't do anything, 
    # but it's placed here so that OTGen and PTGen fit well together

    answers = []

    for i in range(len(inputs) // batch_size): # VERRY HACKY!!! Right now I have this function handling both the case if you want ot for a whole dataset or a single batch
        # Perhaps split that up later.
        if i % 10 == 0 and verbose:
              print("Batch Number: {}".format(i))

        real_vec = encodings[i*batch_size: (i+1)*batch_size]
        fake_vec = inputs[i*batch_size: (i+1)*batch_size]
          
        a = np.ones((batch_size, ))
        b = np.ones((batch_size, ))
          
        M = []
          
        for j in range(batch_size):
            costs = []
            
            for k in range(batch_size):
                costs.append(np.linalg.norm(fake_vec[j] - real_vec[k]))
                #costs.append(np.sum(np.abs(fake_vec[j] - real_vec[k]))) # TODO: Debug the abs & sum & encodings vs real_vec?
            M.append(costs)
            
        M = np.array(M)
        mapping = ot.emd(a, b, M)
          
        for j in range(batch_size):
            index = np.argmax(mapping[j])
            answers.append(real_vec[index])

    answers = np.array(answers)
    return (inputs, answers)

def pt_compute_answers(encodings, batch_size, inputs=None, distr='uniform', ratio=1):
    if inputs is None:
        if distr == 'uniform':
            inputs = np.random.random(size=encodings.shape)
        elif distr == 'normal':
            inputs = np.random.normal(size=encodings.shape)
        else:
            raise Exception("I don't recognize this distribution: {}".format(distr))
    
    answers = []

    for i in range(len(inputs)):
        if i % 100 == 0:
            print('Index: {} out of {}'.format(i, len(inputs)))
        min_dist = 100000
        index = 0
        for j in range(int(len(inputs)*ratio)):
            dist = np.linalg.norm(inputs[i] - encodings[j])
            #dist = np.sum(np.abs(inputs[i] - encodings[j]))
          
            if dist < min_dist:
                min_dist = dist
                index = j
          
        answers.append(encodings[index])

    answers = np.array(answers)

    return(inputs, answers)