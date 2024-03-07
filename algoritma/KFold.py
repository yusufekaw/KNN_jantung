from sklearn.model_selection import KFold
def Split(X_train, y_train):
    # Menginisialisasi objek KFold
    kfold = KFold(n_splits=2, shuffle=False, random_state=None)
    # Lakukan iterasi K-Fold
    for train_index, val_index in kfold.split(X_train):
        X_train_fold1, X_train_fold2 = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold1, y_train_fold2 = y_train.iloc[train_index], y_train.iloc[val_index]
    return X_train_fold1, X_train_fold2, y_train_fold1, y_train_fold2