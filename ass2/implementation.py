import tensorflow as tf

import re
BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 40  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",'a', 'an', 'the', 'actor', 'actress', 'movie', 'cast', 'story', 'plot', 'director', 'film', 'all'
        ,'and', 'are', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'between', 'both', 'by', 'could',
                'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'having', 'he', 'hed',
                'hell', 'hes', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how',
                'i', 'id', 'ill', 'im', 'ive', 'if', 'in', 'into', 'is', 'it', 'its', 'its', 'itself', 'lets', 'me',
                'my', 'has', 'will', 'myself', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
                'ourselves', 'out', 'own', 'she', 'shed', 'shell', 'shes', 'so', 'some', 'such', 'than', 'that',
                'thats', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'theres', 'these', 'they',
                'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those', 'through', 'to', 'too', 'until', 'up', 'was',
                'we', 'wed', 'weve', 'were', 'what', 'whats', 'when', 'whens', 'where', 'wheres', 'which', 'while', 'who',
                'whos', 'whom', 'with', 'would', 'you', 'youd', 'youll', 'youre', 'youve', 'your', 'yours', 'yourself', 'yourselves'
])

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    processed_review = []
    a = review.split()
    for i in range(len(a)):
        a[i] = a[i].lower()

    for i in range(len(a)):
        s = a[i]
        s=re.sub('[^A-Za-z]', '', s)
        if s.isalpha() and len(s) > 2 and s not in stop_words and a[i] not in stop_words :

            processed_review.append(s)



    return processed_review

def transform(input_data):
    data=tf.transpose(input_data,[1,0,2])
    data=tf.reshape(data,[-1,EMBEDDING_SIZE])
    data=tf.split(data,MAX_WORDS_IN_REVIEW)
    return data

def setlstmcell(lstm,dropout_keep_prob):
    lstmcell=tf.contrib.rnn.LSTMCell(lstm, forget_bias=1.0)
    lstmcell=tf.contrib.rnn.DropoutWrapper(cell=lstmcell,output_keep_prob=dropout_keep_prob)
    return lstmcell

def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """





    labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 2], name='labels')
    input_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MAX_WORDS_IN_REVIEW,EMBEDDING_SIZE], name='input_data')
    dropout_keep_prob=tf.placeholder_with_default(0.6,shape=())
    data=transform(input_data)


    lstm=30
    # lstm_fw_cell = tf.contrib.rnn.LSTMCell(lstm, forget_bias=1.0)
    # lstm_bw_cell = tf.contrib.rnn.LSTMCell(lstm, forget_bias=1.0)
    # lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, output_keep_prob=dropout_keep_prob)
    # lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, output_keep_prob=dropout_keep_prob)
    lstm_fw_cell=setlstmcell(lstm,dropout_keep_prob)
    lstm_bw_cell=setlstmcell(lstm,dropout_keep_prob)
    outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, data,
                                                            dtype=tf.float32)


    weight = tf.Variable(tf.truncated_normal([2 * lstm, 2]))
    bias = tf.Variable(tf.random_normal([2]))
    prediction = tf.matmul(outputs[-1], weight) + bias
    correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32), name='accuracy')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels), name='loss')
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)













    return input_data, labels,  dropout_keep_prob,optimizer, accuracy, loss

