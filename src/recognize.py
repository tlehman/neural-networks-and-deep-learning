# web wrapper for MNIST neural network recognizer
import web
import json
import urllib
import cStringIO
from PIL import Image

# from Michael Nielson's book on neural networks and deep learning
from network import *


urls = (
    "/", "index",
    "/(.*)", "recognize"
)
app = web.application(urls, globals())

# Initialize neural network
net = Network([784, 30, 10])
# Load previously learned parameters
net.load_from_pickled_parameters()

class index:
    def GET(self):
        f = open("src/static/index.html", "r")
        output = f.read()
        f.close()
        return output

class recognize:
    def POST(self, name):
        # read in posted base64 data, assume PNG, convert to greyscale
        data = web.data()
        file = cStringIO.StringIO(urllib.urlopen(data).read())
        img = Image.open(file).convert('L')
        # resize to 28x28
        img.thumbnail((28,28), Image.ANTIALIAS)
        # convert to vector
        vec = np.asarray(img).reshape((28*28,1)).astype(float)
        # feed foward through neural network
        digit = net.recognize(vec)
        print digit
        return digit

if __name__ == "__main__":
    app.run()
