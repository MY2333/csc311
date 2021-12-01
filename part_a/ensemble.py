# TODO: complete this file.
from numpy import nan
from knn import *


from utils import *
train_data = load_train_csv("../data")
val_data = load_valid_csv("../data")
test_data = load_public_test_csv("../data")

def bootstrap():
    sparse_matrix = np.full((542, 1774), nan)
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
        if np.isnan(sparse_matrix[user_id - 1][question_id - 1]):
            sparse_matrix[user_id - 1][question_id - 1] = is_correct

    return sparse_matrix, boot_train



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



if __name__ == "__main__":
    sparse_matrix, boot_train = bootstrap()
    knn_predict = knn(sparse_matrix, val_data)
    print(evaluate(val_data,knn_predict))







