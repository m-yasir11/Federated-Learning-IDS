from dataset import preprocess_unsw

X_train, y_train, X_test, y_test, _ = preprocess_unsw(
    "data/UNSW_NB15_training-set.csv",
    "data/UNSW_NB15_testing-set.csv"
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)
print("Label distribution:\n", y_train.value_counts())

