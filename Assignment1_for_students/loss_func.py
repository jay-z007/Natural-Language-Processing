import tensorflow as tf


def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    
    U = true_w
    V = inputs

    V_UT = tf.matmul(V, U, transpose_b=True)
    
    A = tf.diag_part(V_UT)
    B = tf.log(tf.reduce_sum(tf.exp(V_UT), axis=1))

    return tf.subtract(B, A)


def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

    epsilon = 1e-10

    batch_size, embedding_size = inputs.shape
    num_sampled = k = sample.shape[0]
    wc = inputs # [batch_size, embedding_size]
    unigram_prob = tf.convert_to_tensor(unigram_prob)
    
    w0 = tf.reshape(tf.nn.embedding_lookup(weights, labels), [batch_size, embedding_size]) # [batch_size, embedding_size]
    b0 = tf.nn.embedding_lookup(biases, labels) # [batch_size, 1]
    P_w0 = tf.reshape(tf.nn.embedding_lookup(unigram_prob, labels), [batch_size, 1]) # [batch_size, 1]
    
    wx = tf.nn.embedding_lookup(weights, sample) # [num_sampled, embedding_size]
    bx = tf.reshape(tf.nn.embedding_lookup(biases, sample), [num_sampled, 1]) # [num_sampled, 1]
    P_wx = tf.reshape(tf.nn.embedding_lookup(unigram_prob, sample), [num_sampled, 1]) # [num_sampled, 1]
    
    s = tf.matmul(wc, w0, transpose_b=True) # [batch_size, batch_size]
    s = tf.reshape(tf.diag_part(s), [batch_size, 1]) + b0 # [batch_size, 1]
    P_w0_wc = tf.sigmoid(s - tf.log(k*P_w0)) # [batch_size, 1]
    
    s = tf.matmul(wc, wx, transpose_b=True) # [batch_size, num_sampled]
    bx = tf.tile(bx, [1, batch_size]) # [num_sampled, batch_size]
    bx = tf.transpose(bx) # [batch_size, num_sample]
    s += bx
    
    # same as the step above
    P_wx = tf.transpose(tf.tile(P_wx, [1, batch_size])) # [batch_size, num_sampled]
    P_wx_wc = tf.sigmoid(s - tf.log(k*P_wx)) # [batch_size, num_sampled]

    A = tf.log(P_w0_wc + epsilon) # [batch_size, 1]
    B = tf.reshape(tf.reduce_sum(tf.log(1 - P_wx_wc + epsilon), axis=1), [batch_size, 1]) # [batch_size, 1]


    return tf.scalar_mul(-1, tf.add(A, B))




