# web wrapper for MNIST neural network recognizer
import web
import json

# from Michael Nielson's book on neural networks and deep learning
from network import *

urls = (
    '/(.*)', 'recognize'
)
app = web.application(urls, globals())

# Load MNIST data
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Initialize neural network
net = Network([784, 30, 10])
# Load previously learned parameters
net.load_from_pickled_parameters()

# Load trained weights and values (30 epochs, minibatch size of 10, eta=3.0)
# net.SGD(training_data, 30, 10, 3.0, test_data=test_data)


class recognize:
    def POST(self, name):
        data = json.loads(web.data())
        return "This is your img data: %s" % data['img']

if __name__ == "__main__":
    app.run()
