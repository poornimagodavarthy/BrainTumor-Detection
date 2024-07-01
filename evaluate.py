from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import sns


def evaluate_model(model, resized_testing):
    datagen = ImageDataGenerator(rescale=1./255)
    test_generator = datagen.flow_from_directory(
        resized_testing,
        target_size=(512, 512),
        batch_size=64,
        class_mode='categorical',
        shuffle=False
    )

    # Evaluate the loaded model on the test set
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test accuracy: {test_acc}")
    print(f"Test loss: {test_loss}")

    # make predictions
    predictions = model.predict(test_generator)
    predicted_classes= np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    #get true classes:
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Print the classification report
    print("Classification Report:")
    report = classification_report(y_true, predicted_classes, target_names=class_labels)
    print(report)

    # Print the confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(true_classes, predicted_classes)
    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


model_path = '/path/to/Neuro/model.h5'
resized_testing = '/path/to/resized/testing/data'

model = load_model(model_path)
get_report = evaluate_model(model, resized_testing)