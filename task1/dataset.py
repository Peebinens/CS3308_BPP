# Function: Generate the dataset for the 3D bin packing problem
import numpy as np
import random
def rorate_data(data):
    # Rotate the data by angle
    ret = data.copy()
    r_ind = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]  
    for i in range(3) :
        ret[i] = data[r_ind[data[6]][i]]    
    return ret

def derorate_data(data):
    # De-rotate the data by angle
    ret = data.copy()
    r_ind = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]  
    for i in range(3) :
        ret[r_ind[data[6]][i]] = data[i]    
    return ret

def load_data(L, W, H):
    # return a list of items, each item is a list of [L, W, H, x, y, z, r]
    # L, W, H: the size of the project
    # x, y, z: the position of the item
    # r: the rotation of the item
    N = np.random.randint(10, 50) - 1
    I = [[L, W, H, 0, 0, 0, 0]] 
    for i in range(N):
        # pop an item randomly from I by item's volume
        print([item[0]*item[1]*item[2] for item in I])
        pop_idx = random.choices(range(len(I)), weights=[item[0]*item[1]*item[2] for item in I], k=1)[0]
        pop_item = derorate_data(I.pop(pop_idx))
        # choose the axis by the each edge's length
        axis = random.choices(range(3), weights=pop_item[:3], k=1)[0]
        # split the item by the axis and rotate the two new items
        l = pop_item[axis]  
        l1 = np.random.rand()*l
        l2 = l - l1
        split_1 = pop_item.copy()
        split_1[axis] = l1
        split_2 = pop_item.copy()
        split_2[axis] = l2
        split_2[3+axis] = pop_item[3+axis] + l1
        split_1[6] = np.random.randint(0, 5)
        split_2[6] = np.random.randint(0, 5)
        split_1 = rorate_data(split_1)
        split_2 = rorate_data(split_2)
        I.append(split_1)
        I.append(split_2)
    return I

def test_one_data(I):
    # draw the data
    import matplotlib.pyplot as plt 
    print(I)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for item in I:
        dx, dy, dz, x, y, z = derorate_data(item)[:6]
        color = np.random.rand(3)
        ax.bar3d(x, y, z, dx, dy, dz, color=color, shade=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def load_dataset(N):
    # Load the dataset with N samples
    dataset = []
    for i in range(N):
        L = 100
        W = 100
        H = 100
        data = load_data(L, W, H)
        dataset.append(data)
    return dataset

def save_dataset(data, filename):
    # Save the dataset to a file
    with open (filename, 'w') as f:
        for item in data:
            f.write(' '.join([str(i) for i in item]) + '\n')
if __name__ == '__main__':
    dataset = load_dataset(1)
    test_one_data(dataset[0])  
    save_dataset(dataset[0], 'dataset.txt') 
