import tensorflow as tf
tf.random.set_seed(1)
class GenreModel(tf.keras.Model):

    def __init__(self, num_layer, num_units, num_classes):
        super(GenreModel,self).__init__()
        super(GenreModel,self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_length=100,input_dim=10000,output_dim=50)
        self.gru = tf.keras.layers.GRU(num_units, return_sequences=True)
        self.classifier = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax'))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.num_layer = num_layer

    def __call__(self, x):
        xo = self.embedding(x)
        for _ in range(len(self.num_layer)):
            xo = self.gru(xo)
        xo = self.classifier(xo)
        return xo