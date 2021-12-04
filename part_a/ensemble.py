# TODO: complete this file.
from numpy import nan
from knn import *
from item_response import *

from utils import *


train_data = load_train_csv("../data")
val_data = load_valid_csv("../data")
test_data = load_public_test_csv("../data")


def bootstrap():
    boot_matrix = np.full((542, 1774), nan)
    boot_train = {"user_id": [], "question_id": [], "is_correct": []}
    size = len(train_data['user_id'])
    index_lst = np.random.choice(size-1, size)

    for i in index_lst:
        user_id = train_data["user_id"][i]
        question_id = train_data["question_id"][i]
        is_correct = train_data["is_correct"][i]
        boot_train["user_id"].append(user_id)
        boot_train["question_id"].append(question_id)
        boot_train["is_correct"].append(is_correct)
        if np.isnan(boot_matrix[user_id - 1][question_id - 1]):
            boot_matrix[user_id - 1][question_id - 1] = is_correct


    return boot_matrix, boot_train


# user-based collaborative filtering
def knn(matrix, valid_data):
    valid_predict = []
    nbrs = KNNImputer(n_neighbors=11)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    data_size = len(valid_data["user_id"])
    for i in range(data_size):
        user_id = valid_data["user_id"][i]
        question_id = valid_data["question_id"][i]
        valid_predict.append(mat[user_id][question_id])
    return valid_predict


def knn_predict_func(sparse_matrix):
    predict_val = knn(sparse_matrix, val_data)
    val_acc = evaluate(val_data, predict_val)
    predict_test = knn(sparse_matrix, test_data)
    test_acc = evaluate(test_data,predict_test)
    return val_acc, test_acc

def ir_predict_function(theta, beta, data):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a)
    return pred

def ir(boot_train):
    theta, beta, val_acc_lst, train_llh, valid_llh = irt(boot_train, val_data, 0.001, 100)
    val = ir_predict_function(theta, beta, val_data)
    test = ir_predict_function(theta, beta, test_data)
    return val, test

if __name__ == "__main__":
    # get bootstrap data
    boot_matrix, boot_train = bootstrap()

    # knn model
    knn_predict = knn_predict_func(boot_matrix)
    print("Validation for KNN" + str(knn_predict[0]))
    print("Testing for KNN" + str(knn_predict[1]))

    # ir model
    ir_predict = ir(boot_train)
    print("Validation for IR:"+str(ir_predict[0]))
    print("Testing for IR:"+str(ir_predict[1]))







