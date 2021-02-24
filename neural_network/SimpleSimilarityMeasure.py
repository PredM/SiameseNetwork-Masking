import tensorflow as tf

# "as SSM" is necessary due to duplicate class names
from configuration.Enums import SimpleSimilarityMeasure as SSM


class SimpleSimilarityMeasure:

    def __init__(self, sim_type):
        self.sim_type = sim_type

        self.a_weights = None
        self.b_weights = None
        self.a_context = None
        self.b_context = None
        self.w = None

    @tf.function
    def get_sim(self, a, b, a_weights=None, b_weights=None, a_context=None, b_context=None, w=None):

        # assign to class variables so only common parameters must be passed below
        self.a_weights = a_weights
        self.b_weights = b_weights
        self.a_context = a_context
        self.b_context = b_context
        self.w = w

        switcher = {
            SSM.ABS_MEAN: self.abs_mean,
            SSM.EUCLIDEAN_SIM: self.euclidean_sim,
            SSM.EUCLIDEAN_DIS: self.euclidean_dis,
            SSM.COSINE: self.cosine,
        }

        # Get the function from switcher dictionary
        func = switcher.get(self.sim_type)
        return func(a, b)

    def get_weight_matrix(self, a):
        weight_matrix = tf.reshape(tf.tile(self.a_weights, [a.shape[0]]), [a.shape[0], a.shape[1]])
        a_weights_sum = tf.reduce_sum(weight_matrix)
        a_weights_sum = tf.add(a_weights_sum, tf.keras.backend.epsilon())
        weight_matrix = weight_matrix / a_weights_sum

        return weight_matrix

    # Siamese Deep Similarity as defined in NeuralWarp
    # Mean absolute difference (L1, Manhattan Distance)
    @tf.function
    def abs_mean(self, a, b):

        use_weighted_sim = self.a_weights is not None and self.b_weights is not None
        use_additional_sim = self.a_context is not None and self.b_context is not None

        if use_weighted_sim:
            # Note: only one weight vector is used (a_weights) to simulate a retrieval situation
            # where only weights of the case are known
            weight_matrix = self.get_weight_matrix(a)
            #a = a / tf.sqrt(tf.math.reduce_sum(tf.square(a), axis=1, keepdims=True) + 1e-8)
            #b = b / tf.sqrt(tf.math.reduce_sum(tf.square(b), axis=1, keepdims=True) + 1e-8)
            diff = tf.abs(a - b)
            # feature (data stream) weighted distance:
            distance = tf.reduce_mean(weight_matrix * diff)

            if use_additional_sim:
                # calculate context distance
                diff_con = tf.abs(self.a_context - self.b_context)
                distance_con = tf.reduce_mean(diff_con)

                # Combine attribute weighted distance with contextual distance:
                if self.w is None:
                    # Other option define fix w value
                    #self.w = 0.5
                    #distance = self.w * distance + (1 - self.w) * distance_con

                    distance2 = distance + distance_con
                    #tf.print("Sum: ", distance2, "dis_node: ", distance, "dis_graph: ", distance_con)
                    distance = distance2

                else:
                    # Used in iotstream paper
                    # weight both distances
                    distance = self.w * distance + (1 - self.w) * distance_con
                    distance = tf.squeeze(distance)
                    #tf.print("w: ", self.w, "dis1: ",distance, "dis2: ",distance_con)
        else:
            diff = tf.abs(a - b)
            distance = tf.reduce_mean(diff)
        sim = tf.exp(-distance)
        #tf.print("dis1: ",tf.reduce_mean(distance), "dis2: ", tf.reduce_mean(distance_con))
        #tf.print("a: ", a)
        #tf.print("b: ", b)
        #tf.print("Sim: ", sim)
        return sim

    # Euclidean distance (required in contrastive loss function and converted sim)
    @tf.function
    def euclidean_dis(self, a, b):
        # cnn2d, [T,C]
        use_weighted_sim = self.a_weights is not None and self.b_weights is not None
        use_additional_sim = self.a_context is not None and self.b_context is not None

        # Combine attribute weighted distance with contextual distance:
        if use_weighted_sim:
            # Note: only one weight vector is used (a_weights) to simulate a retrieval situation
            # where only weights of the case are known
            weight_matrix = self.get_weight_matrix(a)
            q = a - b
            weighted_dist = tf.sqrt(tf.reduce_sum(weight_matrix * q * q))
            e_dis = weighted_dist
            if use_additional_sim:
                if self.w is None:
                    #tf.print("self.a_context: ", self.a_context)
                    #tf.print("self.b_context: ", self.b_context)

                    # calculate context distance
                    contextual_dis = tf.norm(self.a_context - self.b_context, ord='euclidean')
                    contextual_dis = tf.reduce_mean(contextual_dis)

                    e_dis = contextual_dis + weighted_dist

                    # Other option: weight both distances based on a fixed value w
                    #self.w = 0.5
                    #contextual_dis = contextual_dis + tf.keras.backend.epsilon()
                    #weighted_dist = weighted_dist + tf.keras.backend.epsilon()
                    #distance = self.w * weighted_dist + (1 - self.w) * contextual_dis
                    #e_dis = tf.squeeze(distance)

                    #tf.print("w: ", self.w, "dis1: ",distance, "dis2: ",distance_con)
                else:
                    # weight both distances based on a learned value w
                    #tf.print("w: ", self.w)
                    contextual_dis = tf.norm(self.a_context - self.b_context, ord='euclidean')
                    e_dis = self.w * weighted_dist + (1 - self.w) * contextual_dis
                    e_dis = tf.squeeze(e_dis)
                    # tf.print("w: ", self.w, "dis1: ",distance, "dis2: ",distance_con)
        else:
            e_dis = tf.norm(a - b, ord='euclidean')
        #diff= diff + tf.keras.backend.epsilon()
        #tf.print("diff final:", diff)
        return e_dis

    # Euclidean distance converted to a similarity
    @tf.function
    def euclidean_sim(self, a, b):

        diff = self.euclidean_dis(a, b)
        sim = 1 / (1 + tf.reduce_sum(diff))
        return sim

    @tf.function
    def cosine(self, a, b):
        use_weighted_sim = self.a_weights is not None and self.b_weights is not None
        if use_weighted_sim:
            # source: https://stats.stackexchange.com/questions/384419/weighted-cosine-similarity
            weight_vec = self.a_weights / tf.reduce_sum(self.a_weights)
            normalize_a = tf.nn.l2_normalize(a, 0) * weight_vec
            normalize_b = tf.nn.l2_normalize(b, 0) * weight_vec
            cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b) * weight_vec)
            # cos_similarity = 1-distance.cosine(a.numpy(),b.numpy(),self.a_weights)
        else:
            normalize_a = tf.nn.l2_normalize(a, 0)
            normalize_b = tf.nn.l2_normalize(b, 0)
            cos_similarity = tf.reduce_sum(tf.multiply(normalize_a, normalize_b))
            # tf.print(cos_similarity)

        return cos_similarity
