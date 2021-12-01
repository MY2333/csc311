from sklearn.impute import KNNImputer
from utils import *
import matplotlib.pyplot as plt

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    acc = None
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy: {}".format(acc))
    return acc
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)

    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    k_lst = [1, 6, 11, 16, 21, 26]
    # user-based collaborative filtering
    acc_user = []
    for k in k_lst:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        acc_user.append(acc)
    # Plot and report the accuracy on the validation data as a function of k
    plt.xlabel("k values")
    plt.ylabel("accuracy")
    plt.title("user-based collaborative filtering")
    plt.plot(k_lst, acc_user)
    plt.legend(["validation"])
    plt.savefig("user_acc_valid.png")
    # Choose k that has the highest performace on calidation data.
    # Report the chosen k and the test accuracy
    best_k = k_lst[acc_user.index(max(acc_user))]
    test_acc = knn_impute_by_user(sparse_matrix, test_data, best_k)
    print("Based on student similarity\n")
    print("the best k is %d with accuracy %f\n" % (best_k, test_acc))
    # the best k is 11 with accuracy 0.684166

    # item-based collaborative filtering
    acc_item = []
    for k in k_lst:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        acc_item.append(acc)
    # Plot and report the accuracy on the validation data as a function of k
    plt.xlabel("k values")
    plt.ylabel("accuracy")
    plt.title("item-based collaborative filtering")
    plt.plot(k_lst, acc_item)
    plt.legend(["validation"])
    plt.savefig("item_acc_valid.png")
    # Choose k that has the highest performace on calidation data.
    # Report the chosen k and the test accuracy
    best_k = k_lst[acc_item.index(max(acc_item))]
    test_acc = knn_impute_by_item(sparse_matrix, test_data, best_k)
    print("Based on question similarity\n")
    print("the best k is %d with accuracy %f\n" % (best_k, test_acc))
    # the best k is 21 with accuracy 0.681626
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
