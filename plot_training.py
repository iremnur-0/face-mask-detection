import matplotlib.pyplot as plt

epochs = list(range(1, 11))
train_loss = [27.17, 15.22, 11.45, 10.38, 8.41, 10.77, 7.47, 5.66, 7.53, 6.41]
val_loss =   [4.17,  4.11,  4.53,  2.48, 2.82,  2.56, 2.28, 3.06, 3.40, 2.98]

train_acc = [0.9058, 0.9475, 0.9583, 0.9673, 0.9712, 0.9621, 0.9753, 0.9767, 0.9767, 0.9760]
val_acc =   [0.9279, 0.9279, 0.9211, 0.9581, 0.9513, 0.9648, 0.9581, 0.9530, 0.9413, 0.9547]


plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png")  
plt.show()
