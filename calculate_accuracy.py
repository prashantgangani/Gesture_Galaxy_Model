import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Standardize the data to have zero mean and unit variance
max_length = max(len(item) for item in data_dict['data'])
standardized_data = [np.pad(item, (0, max_length - len(item)), 'constant') if len(item) < max_length else item[:max_length] for item in data_dict['data']]

data = np.asarray(standardized_data)
labels = np.asarray(data_dict['labels'])

print("=== Model Accuracy Analysis ===")
print(f"Total samples: {len(data)}")
print(f"Total classes: {len(set(labels))}")

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42)

print(f"\nTraining samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Calculate basic accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n=== Basic Accuracy ===")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Cross-validation accuracy
print(f"\n=== Cross-Validation Analysis ===")
cv_scores = cross_val_score(model, data, labels, cv=5, scoring='accuracy')
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"Individual CV scores: {cv_scores}")

# Stratified K-Fold for better evaluation on imbalanced data
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratified_scores = cross_val_score(model, data, labels, cv=skf, scoring='accuracy')
print(f"Stratified 5-Fold CV Accuracy: {stratified_scores.mean():.4f} (+/- {stratified_scores.std() * 2:.4f})")

# Detailed classification report
print(f"\n=== Detailed Classification Report ===")
class_names = sorted(set(labels))
report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
print(report)

# Precision, Recall, F1-score for each class
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)

print(f"\n=== Per-Class Metrics ===")
print(f"{'Class':<6} {'Precision':<10} {'Recall':<8} {'F1-Score':<8} {'Support':<8}")
print("-" * 50)
for i, class_name in enumerate(class_names):
    if i < len(precision):
        print(f"{class_name:<6} {precision[i]:<10.4f} {recall[i]:<8.4f} {f1[i]:<8.4f} {support[i]:<8}")

# Macro and Micro averages
macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='micro', zero_division=0)

print(f"\n=== Average Metrics ===")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall: {macro_recall:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"Micro Precision: {micro_precision:.4f}")
print(f"Micro Recall: {micro_recall:.4f}")
print(f"Micro F1-Score: {micro_f1:.4f}")

# Confusion Matrix
print(f"\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (first 10x10 for readability):")
print(cm[:10, :10])

# Feature importance
print(f"\n=== Feature Importance ===")
feature_importance = model.feature_importances_
print(f"Top 10 most important features:")
top_features = np.argsort(feature_importance)[-10:][::-1]
for i, feature_idx in enumerate(top_features):
    print(f"{i+1}. Feature {feature_idx}: {feature_importance[feature_idx]:.4f}")

# Model performance summary
print(f"\n=== Model Performance Summary ===")
print(f"Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"Micro F1-Score: {micro_f1:.4f}")

# Save the trained model
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
print(f"\nModel saved to 'model.p'")

# Create visualization of confusion matrix
plt.figure(figsize=(12, 10))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"\nConfusion matrix saved as 'confusion_matrix.png'")

# Create class distribution plot
plt.figure(figsize=(15, 6))
class_counts = [sum(1 for label in labels if label == cls) for cls in class_names]
plt.bar(class_names, class_counts)
plt.title('Class Distribution in Dataset')
plt.xlabel('Classes')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
print(f"Class distribution plot saved as 'class_distribution.png'")
