import tensorflow as tf
import numpy as np
import json
import os

from random import randint

from modules.LoadMNIST import load_mnist,encode_data


class DigitAI:
    def __init__(self):
        self.data_path = "Data/"
        self.AI_path = "Brains/"
        self.AI_name = "MNIST_Digit_Recognizer"

        # Strings for construction of AI answers
        self.primer = {
            "certain": ["I'm sure that is ","I know! That's ", "Clearly, that is "],
            "confident": ["I think that is ", "I believe that's ","I would say that's "],
            "unsure": ["That could be ", "Maybe that's ", "I'm guessing that's "],
            "clueless": ["I have no clue..","That could be anything..","Nope, I got nothing.."]
            }
        # Names of numbers
        self.digit = {
            0: "zero",
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine"}

        # Default AI options:
        # Only used when creating a new model.
        # After model creation, options are read from \Brains\[AI name]\options.json -file
        # This allows multiple different models stored in \Brains\
        options_fp =  self.AI_path+self.AI_name+"/options.json"
        
        default_options = {
            "learn_rate" : 5e-4,
            "value_hot" : 100,                            # Numerical value of of hot encoded vector
            "use_crossentropy" : False,     # True: AI uses crossentropy objective function, False: objective function is mean-square error
            "print_decimals" : 0                    # Print this many decimals in AI prediction print outs
            }
            
            

        # Check if model foldel exists, if not it will be created.
        if not os.path.exists(self.AI_path+self.AI_name):
            print("Making new folder for the model")
            os.makedirs(self.AI_path+self.AI_name)
            
         # Load options file if available
        try:
            with open(options_fp,"r") as file:
                self.options=json.load(file)
        except FileNotFoundError:
            print("\n No options file was found.".format(options_fp))
            self.options = dict(default_options)
            with open(options_fp,"w") as file:
                json.dump(self.options,file)
            print("New options file created: {}".format(options_fp))
         
        # Load AI model if available, otherwise create a new one
        try:
            self.core = tf.keras.models.load_model(self.AI_path+self.AI_name)
        except IOError:
            self.core = self._define_core_AI()
            self._train_AI()
            self.core.save(self.AI_path+self.AI_name)
        print("\n")
        self.core.summary()
        print("\nOptions:")
        print(self.options)
        

    def _load_data(self,test = False):
        """
        Retrieves and stores data from the MNIST-files
        """
        if not test:
            self.train_x = load_mnist("train images",self.data_path)
            train_labels = load_mnist("train labels",self.data_path) 
            self.train_y = encode_data(train_labels,self.options["value_hot"])
    
        self.test_x = load_mnist("test images",self.data_path)    
        test_labels = load_mnist("test labels",self.data_path)
        self.test_y = encode_data(test_labels,self.options["value_hot"])



    def _define_core_AI(self):
        """
        Core funtionality of MNIST digit recognition AI
        Takes in array of 28 x 28 cells (MNIST data size)
        Returns identification in one hot encoded vector
        """
        inputs = tf.keras.layers.Input(shape=(28,28,1))

        #conv1 = tf.keras.layers.Conv2D(filters=32,kernel_size = (5,5),activation="relu",kernel_initializer="he_normal")(inputs) #padding="same",
        #pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
        #conv2 = tf.keras.layers.Conv2D(filters=32,kernel_size = (3,3),activation="relu",kernel_initializer="he_normal")(pool1)
        #pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
        flat = tf.keras.layers.Flatten(input_shape = (28,28,1))(inputs)
        dens1 = tf.keras.layers.Dense(units=392,activation ="relu")(flat)
        drop1 = tf.keras.layers.Dropout(0.2, input_shape=(392,))(dens1)
        digits = tf.keras.layers.Dense(units=10,activation ="softmax")(drop1)        

        if self.options["use_crossentropy"]:
            loss_fn = tf.keras.losses.CategoricalCrossentropy()
        else:
            digits = tf.keras.layers.Dense(units=10,activation ="relu")(drop1)
            loss_fn = "mse"            


        model = tf.keras.models.Model(inputs=inputs,outputs=digits,name=self.AI_name)
        
        model.compile(
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.options["learn_rate"]),
            loss = loss_fn,
            metrics=["accuracy"]
            )
        return model


    def _train_AI(self):
        """
        Retrieves Data, and trains core AI
        """
        self._load_data()
        self.core.fit(self.train_x, self.train_y,batch_size=100, epochs=6)
        self.core.evaluate(self.test_x,  self.test_y, verbose=2)


    def test(self):
        """
        Retrieves MNIST test set and run model ealution on it
        """
        self._load_data(test=True)
        print("\nRunnng {} evaluation on MNIST test dataset:".format(self.AI_name))
        self.core.evaluate(self.test_x,  self.test_y, verbose=2)

        

    def recognize(self,img):
        """
        Run digit recognition on input image:
        Args:
            img: numpy array of shape (1,28,28,1)
        Return: 
            (prediction:list, level:str, comment: str) tuple of numpy array of digit scores, prediction confidence estimate and a string comment
        """

        output = self.core.predict(img)                                             # Get Model output
             
        predictions = np.around(output[0],decimals=self.options["print_decimals"])                # Round output to printable accuracy
        level, comment = self._banter(output[0])                            # Get verbal estimates of prediction
        
        return (predictions,level,comment)



    def _banter(self,arr):
        """
        Analyse the model prediction and formulate comment string based on it.
        """
        numbers = []
        level = None

        value_hot = self.options["value_hot"]

        for n,score in enumerate(arr):
            if score > 0.75*value_hot:
                level = "certain"
                numbers.append((n, score))

        if not level:
            for n,score in enumerate(arr):
                if score > 0.60*value_hot:
                    level = "confident"
                    numbers.append((n, score))

        if not level:
            for n,score in enumerate(arr):
                if score > 0.25*value_hot:
                    level = "unsure"
                    numbers.append((n, score))

        if not level:
            level = "clueless"

        
        # Arrange found numbers from highest score to lowest
        numbers.sort(key=lambda score: score[1], reverse = True)
        
        # Get random number for comments
        rand = randint(0,2)
        
        # Generate start of the comment
        comm_list = [self.primer[level][rand]]

        # add found numbers to the comment
        if numbers:
            for n,number in enumerate(numbers):
                comm_list.append(self.digit[number[0]])
                if n < len(numbers)-2:
                    comm_list.append(", ")
                elif n == len(numbers) - 2:
                    comm_list.append(" or ")
        comm_list.append(".")
        comment = "".join(comm_list)
        return level, comment




if __name__ == "__main__":
    AI = DigitAI()
    AI.test()


