import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def computeF1Score(epochs, folder):

    test_f1 = []
    val_f1 = []
    test_acc = []
    val_acc = []

    # load in true values
    test_true = np.load(folder+'test_true.npy')
    val_true = np.load(folder+'val_true.npy')

    # process ground truth labels
    y_test = np.argmax(test_true, axis=2)
    y_val = np.argmax(val_true, axis=2)
    # load in data
    for i in range(epochs):
        # print(i)
        # predict on test
        test_pred = np.load(folder+'test_pred'+str(i)+'.npy')
        # get probability predictions
        y_pred = np.argmax(test_pred, axis=2)
        test_f1 += [f1_score(y_test, y_pred)]
        test_acc += [accuracy_score(y_test, y_pred)]

        # predict on validation set
        val_pred= np.load(folder+'val_pred'+str(i)+'.npy')
        y_pred = np.argmax(val_pred, axis=2)
        val_f1 += [f1_score(y_val, y_pred)]
        val_acc += [accuracy_score(y_val, y_pred)]

    # turn into arrays
    test_f1 = np.array(test_f1)
    val_f1 = np.array(val_f1)
    test_acc = np.array(test_acc)
    val_acc = np.array(val_acc)

    np.save(folder+'test_f1.npy', test_f1)
    np.save(folder+'val_f1.npy', val_f1)
    np.save(folder+'test_acc.npy', test_acc)
    np.save(folder+'val_acc.npy', val_acc)

    # save best f1 score and epoch # to a text file
    with open(folder+'best_f1_epochs.txt', 'w') as f:
        f.write('val f1:'+str(max(val_f1)))
        f.write('at epoch:' + str(np.where(val_f1 == max(val_f1))[0]))
        f.write('\n')
        f.write('test f1:'+str(max(test_f1)))
        f.write('at epoch:' +str(np.where(test_f1 == max(test_f1))[0]))
        f.write('\n')
        f.write('val acc:' + str(max(val_acc)))
        f.write('at epoch:' + str(np.where(val_acc == max(val_acc))[0]))
        f.write('\n')
        f.write('test acc:' + str(max(test_acc)))
        f.write('at epoch:' + str(np.where(test_acc == max(test_acc))[0]))

    # plot f1 score comparison
    plt.plot(range(len(val_f1)), val_f1, color='b', label='val')
    plt.plot(range(len(test_f1)), test_f1, color='g', label='test')
    plt.ylim(0,1)
    plt.legend()
    plt.title('f1-score comparison')
    plt.savefig(folder+'val1-comp.png')
    plt.close()

    plt.plot(range(len(val_acc)), val_acc, color='b', label='val')
    plt.plot(range(len(test_acc)), test_acc, color='g', label='test')
    plt.ylim(0, 1)
    plt.legend()
    plt.title('f1-score comparison')
    plt.savefig(folder + 'val1-acc-comp.png')
    plt.close()

    return

def main():
    '''
    :param epochs: # of epochs the model was trained for, so we can compute the F1 score for each epoch - with the early stopping criteria, 
    this is not the same for each model
    :param folder: folder where model output is saved

    During model training, we save the model predictions and the ground truth values for each of the validation sets. After training we can compute
    the f1 score for each class to determine best f1 and the epoch at which that f1 score occurs and also plot the f1 scores so we can look at trends. 
    '''
    epochs = int(sys.argv[1])
    folder = sys.argv[2]

    computeF1Score(epochs, folder)

    return

if __name__ == "__main__":
    main()