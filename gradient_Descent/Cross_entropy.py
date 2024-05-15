import numpy as np

def binary_cross_entropy(x,w,g):

    z = np.dot(x,w)
    y = 1/(1 + np.exp(-z))
    # z= np.clip(z, 1e-15, 1 - 1e-15)
    
    # loss = -np.mean(g * np.log(z) + (1 - g) * np.log(1 - z)) 
    # return loss, z
    loss = -np.mean(g * np.log(y) + (1 - g) * np.log(1 - y)) 
    return loss, y


def gradient_descent(x,w, g, iterations=100, learning_step=1/3,epsilon = 1e-3):
    # Initialising
    iterations=iterations
    h=learning_step
    f_value = []
    weights = []
    previous_f = None

    
    
    #Estimating optimal parameters
    for i in range(iterations):
        funct, z = binary_cross_entropy(x, w, g)
        # funct, z = binary_cross_entropy(x, w, g)
        # n = x.shape[0]
        

        # if previous_f and (previous_f - funct) < 0:
        #     break

        previous_f=funct

        

        f_value.append(funct)
        weights.append(w)
        
        gradient = np.dot(x.T, (z- g)) / len(g)

        w = w - (h * gradient)
        # w=np.clip(w,0,1)
        # w=1/(1+(np.exp(-w)))
        # w = np.modf(w)
        # print(w)
        f_value.append(funct)
        weights.append(w)  

        # if abs(previous_f) <= epsilon:
        if np.all(np.abs(previous_f) <= epsilon):
            break     
        
        
        print(f"Iteration {i+1}: objective function {funct}")
        
        print(f"the value of w is:{w}")
        

        # print(f"value of f_gradient at {i+1} iteration: {np.mean(np.square(y- g))}")
        

    
    return w, f_value, weights

if __name__ == "__main__":
    x = np.array([[1, 2, 3], [4, 5, 6], [7,8 , 9]])
    w = np.ones((3,1))
    g = np.array([0.1, 0.2, 0.5])

    # print(g.shape)

    # funct = f(x, w, g)
    # print(funct)
    # print(np.gradient(funct))

    weight, f_value, weights = gradient_descent(x, w, g, learning_step= 0.1)
