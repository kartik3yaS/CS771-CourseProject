{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "NTTi2OM9mHGr",
        "outputId": "7f6fa26e-4dc4-4882-ac47-7c48d7a9844e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "import sklearn\n",
        "from scipy.linalg import khatri_rao\n",
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.svm import LinearSVC\n",
        "\n",
        "from google.colab import drive\n",
        "drive.mount('/content/drive')\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7t9mrl372ASv"
      },
      "outputs": [],
      "source": [
        "train_data = np.loadtxt('/content/drive/My Drive/collab_file_upload/train.dat')\n",
        "\n",
        "X_train = train_data[:, :-1]\n",
        "y_train = train_data[:, -1]\n",
        "\n",
        "test_data = np.loadtxt('/content/drive/My Drive/collab_file_upload/test.dat')\n",
        "\n",
        "X_test = test_data[:, :-1]\n",
        "y_test = test_data[:, -1]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "tcQsYN1QyUBv",
        "outputId": "f657a848-7301-4a20-d868-ae87bea50b32"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 0.9929\n"
          ]
        }
      ],
      "source": [
        "def my_map(X):\n",
        "    # Convert the elements to -1 and 1\n",
        "    X = 1 - 2 * X\n",
        "\n",
        "    n = X.shape[1]\n",
        "    ones_column = np.ones((X.shape[0], 1))\n",
        "    X = np.hstack((X, ones_column))\n",
        "\n",
        "    for row in X:\n",
        "        for i in range(n - 2, -1, -1):\n",
        "            row[i] = row[i + 1] * row[i]\n",
        "\n",
        "    n_features = X.shape[1]\n",
        "    triu_indices = np.triu_indices(n_features, k=1)\n",
        "    new_vector = X[:, triu_indices[0]] * X[:, triu_indices[1]]\n",
        "\n",
        "    return new_vector\n",
        "\n",
        "def my_fit(X_train, y_train):\n",
        "\n",
        "    X_train = my_map(X_train)\n",
        "    # model = LogisticRegression(max_iter=10000,penalty=\"l1\")\n",
        "    model = LinearSVC(max_iter=10000, penalty = \"l1\", dual = False)\n",
        "    model.fit(X_train, y_train)\n",
        "\n",
        "    # Extract the learned weights and bias\n",
        "    w = model.coef_[0]\n",
        "    b = model.intercept_[0] if model.fit_intercept else 0\n",
        "\n",
        "    return w, b\n",
        "\n",
        "w, b = my_fit(X_train, y_train)\n",
        "\n",
        "feat = my_map(X_test)\n",
        "# Make predictions using the learned model\n",
        "predictions = np.dot(feat, w)+b\n",
        "predictions_binary = np.where(predictions > 0, 1, 0)\n",
        "\n",
        "# Calculate accuracy\n",
        "accuracy = accuracy_score(y_test, predictions_binary)\n",
        "\n",
        "print(\"Accuracy:\", accuracy)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "M5bT_aI6--O7"
      },
      "source": [
        "rough *work*\n",
        "❌\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "dwXKVQOZlJSq"
      },
      "outputs": [],
      "source": [
        "# from sklearn.metrics import hinge_loss\n",
        "# hinge_loss_value = hinge_loss(y_test, predictions)\n",
        "# print(\"Hinge Loss:\", hinge_loss_value)\n",
        "\n",
        "\n",
        "\n",
        "# misclassified_samples = (predictions_binary != y_test).sum()\n",
        "# error_rate = misclassified_samples / len(y_test)\n",
        "\n",
        "# print(\"Error Rate:\", error_rate)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "hz3wt7VxhYYu"
      },
      "outputs": [],
      "source": [
        "# print(w.shape[0])\n",
        "# # print(w)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0tzag_kahdDj"
      },
      "outputs": [],
      "source": [
        "# print(b)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "oQqom3s_nO1C"
      },
      "outputs": [],
      "source": [
        "from sklearn.decomposition import PCA\n",
        "X_train1 = my_map(X_train)\n",
        "X_test1 = my_map(X_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8Q97iDtl8PFU"
      },
      "outputs": [],
      "source": [
        "# # prompt: i want to use PCA\n",
        "\n",
        "# from sklearn.decomposition import PCA\n",
        "# X_train1 = my_map(X_train)\n",
        "# X_test1 = my_map(X_test)\n",
        "\n",
        "# # Create a PCA object with the desired number of components\n",
        "# pca = PCA(n_components=400)\n",
        "\n",
        "# # Fit the PCA model to the training data\n",
        "# pca.fit(X_train1)\n",
        "\n",
        "# # Transform the training and test data using PCA\n",
        "# X_train_pca = pca.transform(X_train1)\n",
        "# X_test_pca = pca.transform(X_test1)\n",
        "\n",
        "# # Fit the logistic regression model to the transformed data\n",
        "# model = LogisticRegression(max_iter=10000, penalty=\"l2\")\n",
        "# model.fit(X_train_pca, y_train)\n",
        "\n",
        "# # Make predictions using the learned model\n",
        "# predictions = model.predict(X_test_pca)\n",
        "\n",
        "# # Calculate accuracy\n",
        "# accuracy = accuracy_score(y_test, predictions)\n",
        "\n",
        "# print(\"Accuracy:\", accuracy)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "476sJFU2mjkU",
        "outputId": "ead9cbd6-9f53-49bc-b6ad-bb70e2c79c37"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "50\n",
            "100\n",
            "150\n",
            "200\n",
            "250\n",
            "300\n",
            "350\n",
            "400\n",
            "450\n",
            "500\n",
            "Maximum Accuracy: 0.9907, Best n_components: 528\n"
          ]
        }
      ],
      "source": [
        "from sklearn.decomposition import PCA\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score\n",
        "import numpy as np\n",
        "\n",
        "# Initialize variables to store maximum accuracy and corresponding n_components\n",
        "max_accuracy = 0\n",
        "best_n_components = 0\n",
        "\n",
        "# Define the maximum dimension (max_dim) you want to iterate over\n",
        "max_dim = 528\n",
        "\n",
        "# Assuming you have X_train1, X_test1, y_train, and y_test defined\n",
        "\n",
        "for n_components in range(1, max_dim + 1):\n",
        "    # Create a PCA object with the desired number of components\n",
        "    pca = PCA(n_components=n_components)\n",
        "\n",
        "    # Fit the PCA model to the training data\n",
        "    pca.fit(X_train1)\n",
        "\n",
        "    # Transform the training and test data using PCA\n",
        "    X_train_pca = pca.transform(X_train1)\n",
        "    X_test_pca = pca.transform(X_test1)\n",
        "\n",
        "    # Fit the logistic regression model to the transformed data\n",
        "    model = LogisticRegression(max_iter=10000, penalty=\"l2\")\n",
        "    model.fit(X_train_pca, y_train)\n",
        "\n",
        "    # Make predictions using the learned model\n",
        "    predictions = model.predict(X_test_pca)\n",
        "\n",
        "    # Calculate accuracy\n",
        "    accuracy = accuracy_score(y_test, predictions)\n",
        "\n",
        "    # Check if the current accuracy is higher than the maximum accuracy\n",
        "    if accuracy > max_accuracy:\n",
        "        max_accuracy = accuracy\n",
        "        best_n_components = n_components\n",
        "\n",
        "    if n_components%50==0:\n",
        "        print(n_components)\n",
        "\n",
        "# Print the maximum accuracy and corresponding n_components\n",
        "print(f\"Maximum Accuracy: {max_accuracy}, Best n_components: {best_n_components}\")\n"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}