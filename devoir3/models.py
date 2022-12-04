import nn
import numpy

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        return nn.DotProduct(x,self.get_weights())
    
    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        if nn.as_scalar(self.run(x))>=0:
            return 1
        return -1
        
    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 1 ***"
        # Itére sur le dataSet jusqu'a qu'il n'est aucune erreur de prédiction.
        while True: 
            classificationErreur = False
            for x, y in dataset.iterate_once(1) :
                # Si erreur de predictrion on update le poid W.
                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(x,nn.as_scalar(y))
                    classificationErreur = True 
            # Si aucune erreur de predictrion dans le dataSet est trouver on arrete l'entrainement.
            if not classificationErreur:
                break

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        self.layerDimensions = [300, 300]
        self.miniBatchSize = 1
        self.learningRate = 0.1  
        self.numberOfLayers = 2
        self.hiddenLayers = []
    
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        y = x
        for layer in self.hiddenLayers:        
            # Un réseau de neurones à deux couches y = W2*ReLU(W1*X+b1)+b2
            y = nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(y, layer[0]), layer[1])), layer[2]), layer[3])
        return y
    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        return nn.SquareLoss(self.run(x), y)
   
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 2 ***"
        # Sous-ensemble d'entrainement qui correspond a 10% des données. 
        self.miniBatchSize = int(0.1*dataset.x.shape[0])
        # Verifie que la taille totale du jeu de données soit divisible par la taille du mini-batch.
        while len(dataset.x) % self.miniBatchSize != 0:
            self.miniBatchSize += 1
        self.hiddenLayers = [
            ([
                nn.Parameter(dataset.x.shape[1], self.layerDimensions[i]),
                nn.Parameter(1,self.layerDimensions[i]),
                nn.Parameter(self.layerDimensions[i], dataset.x.shape[1]),
                nn.Parameter(1,1)
            ])
            for i in range(self.numberOfLayers)
        ]
        # Itére sur le dataSet jusqu'a que la moyen des erreur quadratique est inferieur ou egale a 0.02.
        while True:
            losses = []
            for x, y in dataset.iterate_once(self.miniBatchSize):
                loss = self.get_loss(x,y)
                # Construit une lists des paramètres.
                parameterList = []
                for layer in self.hiddenLayers:
                    for parameter in layer:
                        parameterList.append(parameter)
                # Calcule le gradient de chaque paramètre et mes a jour le réseau.
                gradients = nn.gradients(loss, parameterList)
                for i in range(len(parameterList)):
                    param = parameterList[i]
                    param.update(gradients[i], -self.learningRate)
                losses.append(nn.as_scalar(loss))
            # Si la moyen des erreur quadratique est inferieur ou egale a 0.02 on arrete l'entrainement.
            if numpy.mean(losses) <= 0.02:
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """

    def __init__(self):
        # Initialize your model parameters here
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        self.learningRate = 0.09
        self.batch_size = 1
        self.hiddenLayers = [
            [
                nn.Parameter(784, 150),
                nn.Parameter(1, 150),
                nn.Parameter(150, 784),
                nn.Parameter(1,784)
            ],
            [
                nn.Parameter(784, 100),
                nn.Parameter(1, 100),
                nn.Parameter(100, 10),
                nn.Parameter(1,10)
            ],
        ]
    
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        y = x
        for layer in self.hiddenLayers:
            # Un réseau de neurones à deux couches y = W2*ReLU(W1*X+b1)+b2
            y = nn.AddBias(nn.Linear(nn.ReLU(nn.AddBias(nn.Linear(y, layer[0]), layer[1])), layer[2]), layer[3])
        return y
    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        return nn.SoftmaxLoss(self.run(x), y)      
   
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** TODO: COMPLETE HERE FOR QUESTION 3 ***"
        # Sous-ensemble d'entrainement qui correspond a 0.5% des données.
        self.batch_size = int(0.005 * dataset.x.shape[0])
        # Verifie que la taille totale du jeu de données soit divisible par la taille du mini-batch.
        while len(dataset.x) % self.batch_size != 0:
            self.batch_size += 1
        # Itére sur le dataSet jusqu'a atteindre une précision d’au moins 97%.
        while True:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x,y)
                # Construit une lists des paramètres.
                parameterList = []
                for layer in self.hiddenLayers:
                    for parameter in layer:
                        parameterList.append(parameter)
                # Calcule le gradient de chaque paramètre et mes a jour le réseau.
                gradients = nn.gradients(loss, parameterList)
                for i in range(len(parameterList)):
                    parameter = parameterList[i]
                    parameter.update(gradients[i], -self.learningRate)
            # Si on attiend une précision d’au moins 97% on arrete l'entrainement.
            if dataset.get_validation_accuracy() > 0.97:
                break