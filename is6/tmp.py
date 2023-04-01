# Распознавание тестовой выборки
pred = model.predict(train_images)
pred = np.argmax(pred, axis=1)

print(pred.shape)

print(pred[:20])
print(train_labels)

# Выделение неверных вариантов
mask = pred == train_labels
print(mask[:10])

images_false = train_images[~mask]
labels_false = pred[~mask]

print(images_false.shape)