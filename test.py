import model1

twolayermodel =model1.Two_Layer_Net()
test_images, test_labels = model1.load_mnist('./minist', kind='t10k')
file = './bestmodel.npz'
twolayermodel.load_model(file)

Best_accuracy = (twolayermodel.predict(test_images) == test_labels).mean()
print('Best Accuracy:', Best_accuracy)