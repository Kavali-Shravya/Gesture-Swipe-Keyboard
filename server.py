'''
You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.
You can also import packages you want.
But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.
'''

from flask import Flask, request
from flask import render_template
import time
import json

# IMPORTS
from sklearn import metrics
from scipy import interpolate
import numpy as np

app = Flask(__name__)

numSamplePoints = 100

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


def generate_sample_points(points_X, points_Y):
    sample_points_X, sample_points_Y = [], []

    # if template has number of sample points
    if len(points_X) == numSamplePoints:
        return points_X, points_Y

    # Calculate the cumulative distances of all euclidean distance between consecutive points
    cumulative_distance = np.cumsum(np.sqrt(np.ediff1d(points_X, to_begin=0) ** 2 + np.ediff1d(points_Y, to_begin=0) ** 2))

    # Interpolating points
    X = interpolate.interp1d(cumulative_distance / cumulative_distance[-1], points_X)
    Y = interpolate.interp1d(cumulative_distance / cumulative_distance[-1], points_Y)

    return X(np.linspace(0, 1, numSamplePoints)), Y(np.linspace(0, 1, numSamplePoints))

# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)

L = 200

twidth = np.max(template_sample_points_X, axis=1) - np.min(template_sample_points_X, axis=1)
theight = np.max(template_sample_points_Y, axis=1) - np.min(template_sample_points_Y, axis=1)

scaling_matrix = np.diag(L / np.maximum(1, np.max(np.array([twidth, theight]), axis=0)))
scaled_template_points_X = np.matmul(scaling_matrix, template_sample_points_X)
scaled_template_points_Y = np.matmul(scaling_matrix, template_sample_points_Y)

scaled_template_centroid_X, scaled_template_centroid_Y = np.mean(scaled_template_points_X, axis=1), \
                                                         np.mean(scaled_template_points_Y, axis=1)
tx, ty = 0 - scaled_template_centroid_X, 0 - scaled_template_centroid_Y

translation_matrix_X = np.reshape(tx, (-1, 1))
translation_matrix_Y = np.reshape(ty, (-1, 1))
normalized_template_sample_points_X = translation_matrix_X + scaled_template_points_X
normalized_template_sample_points_Y = translation_matrix_Y + scaled_template_points_Y


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y, validIndexs = [], [], [], []
    threshold = 25

    xfirstIndex = gesture_points_X[0]
    yfirstIndex = gesture_points_Y[0]
    xlastIndex = gesture_points_X[-1]
    ylastIndex = gesture_points_Y[-1]
    gStartPoint = np.array([xfirstIndex, yfirstIndex])
    gEndPoint = np.array([xlastIndex, ylastIndex])

    startPoints, lastPoints = [], []

    for i in range(len(template_sample_points_X)):
        startPoints.append([template_sample_points_X[i][0], template_sample_points_Y[i][0]])
        lastPoints.append([template_sample_points_X[i][-1], template_sample_points_Y[i][-1]])

    # Calculate euclidean distances between start points of gesture and templates
    start_distances = metrics.pairwise.euclidean_distances(
        np.reshape(gStartPoint, (1, -1)),
        startPoints)
    start_distances = start_distances[0]

    # Calculate euclidean distances between end points of gesture and templates
    end_distances = metrics.pairwise.euclidean_distances(
        np.reshape(gEndPoint, (1, -1)),
        lastPoints)
    end_distances = end_distances[0]

    # appending index, words, templateX points and templateY points which are less then the threshold
    for i in range(len(start_distances)):
        if start_distances[i] + end_distances[i] < threshold:
            validIndexs.append(i)
            valid_words.append(words[i])
            valid_template_sample_points_X.append(template_sample_points_X[i])
            valid_template_sample_points_Y.append(template_sample_points_Y[i])

    return validIndexs, valid_words, valid_template_sample_points_X, valid_template_sample_points_Y


def get_shape_scores(valid_indices,
                     gesture_sample_points_X,
                     gesture_sample_points_Y,
                     valid_template_sample_points_X,
                     valid_template_sample_points_Y):
    shape_scores = 0
    L = 200

    width = np.max(gesture_sample_points_X) - min(gesture_sample_points_X)
    height = np.max(gesture_sample_points_Y) - min(gesture_sample_points_Y)

    if height == 0 and width == 0:
        scalingValue = 0
    else:
        scalingValue = L / max(width, height)

    scaledGPoints = np.matmul(np.array([[scalingValue, 0],[0, scalingValue]]),
                                      np.array([gesture_sample_points_X, gesture_sample_points_Y]))

    scaledGCentriodX, scaledGCentriodY = np.mean(scaledGPoints[0]), np.mean(scaledGPoints[1])

    # Translate the points
    translation_matrix = np.array([[0 - scaledGCentriodX],
                                   [0 - scaledGCentriodX]])

    normalized_gesture_sample_points = translation_matrix + scaledGPoints
    xScaled = (normalized_template_sample_points_X[valid_indices] - np.reshape(normalized_gesture_sample_points[0], (1, -1))) ** 2
    yScaled = (normalized_template_sample_points_Y[valid_indices] - np.reshape(normalized_gesture_sample_points[1], (1, -1))) ** 2
    shape_scores = np.sum((xScaled + yScaled) ** 0.5, axis=1) / numSamplePoints

    return shape_scores


def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    location_scores = np.zeros((len(valid_template_sample_points_X)))
    radius = 15

    # Calculate alphas for location score
    alphas = np.zeros((numSamplePoints))
    mid_point = numSamplePoints // 2
    for i in range(mid_point):
        x = i / 2450
        alphas[mid_point - i - 1], alphas[mid_point + i] = x, x

    gesture_points = []
    for i in range(numSamplePoints):
        gesture_points.append([gesture_sample_points_X[i], gesture_sample_points_Y[i]])

    # For each template
    for i in range(len(valid_template_sample_points_X)):
        template_points = []
        for j in range(numSamplePoints):
            template_points.append([valid_template_sample_points_X[i][j], valid_template_sample_points_Y[i][j]])

        # calcuating euclidean_distances between gesture and template points
        distances = metrics.pairwise.euclidean_distances(gesture_points, template_points)

        if np.any(np.min(distances, axis=1) > radius) or np.any(np.min(distances, axis=0) > radius):

            location_scores[i] = np.sum(np.multiply(alphas, np.diagonal(distances)))

    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    shape_coef = 0.1
    location_coef = 1 - shape_coef
    integration_scores = shape_coef * shape_scores + location_coef * location_scores
    return integration_scores

def get_best_word(valid_words, integration_scores):
    suggestion = ""
    best_word = 'the'
    n = 4

    sortedIndex = np.argsort(np.array(integration_scores))
    word_int_score_dict = {}
    for i in range(n):
        word_int_score_dict[valid_words[sortedIndex[i]]] = integration_scores[sortedIndex[i]]

    final_score = float('inf')

    for word, int_score in word_int_score_dict.items():
        if (final_score > int_score * (1 - probabilities[word])):
            final_score = int_score * (1 - probabilities[word])
            best_word = word

    if best_word == " ":
        return "No Match Found"

    word_int_score_dict.pop(best_word)

    suggestion = ' , '.join(list(word_int_score_dict.keys()))

    return best_word, suggestion


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    # gesture_points_X = [gesture_points_X]
    # gesture_points_Y = [gesture_points_Y]

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    validIndices, validWords, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)

    best_word = "Word not found"
    suggestions = ""
    end_time = 0
    if len(validWords) != 0:
        shapeScores = get_shape_scores(validIndices, gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

        locationScores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

        integrationScores = get_integration_scores(shapeScores, locationScores)

        best_word, suggestions = get_best_word(validWords, integrationScores)

        end_time = time.time()
    print('suggestions  ', suggestions)

    return '{"best_word": "' + best_word + '","other_suggestions":"'+suggestions+'", "elapsed_time": "' + str(round((end_time - start_time) * 1000, 5)) + ' ms"}'


if __name__ == "__main__":
    app.run()