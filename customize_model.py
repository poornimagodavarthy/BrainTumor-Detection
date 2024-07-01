from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam

#Get MobileNetV2 Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(512, 512, 3))

#Add Custom Layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

#Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

#Compile Model
model.compile(optimizer=Adam(learning_rate=0.000066),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
