import numpy as np
import pickle
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

def train_evaluate_models():
    # Create output directory
    os.makedirs('output/models', exist_ok=True)
    
    print("Loading processed data...")
    # Load processed data
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    # Load feature names for interpretation
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    print("Defining models...")
    # Define models
    models = {
        'LogisticRegression': LogisticRegression(
            random_state=42, 
            class_weight='balanced',
            max_iter=1000
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'NeuralNetwork': MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=1000,
            random_state=42
        )
    }
    
    # Results dictionary
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(model, f'models/{name}_model.pkl')
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        # Print metrics
        print(f"\nResults for {name}:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC:       {auc:.4f}")
        
        # Classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'output/models/{name}_confusion_matrix.png')
        plt.close()
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.savefig(f'output/models/{name}_roc_curve.png')
        plt.close()
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            
            # Get feature importances
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Plot top 20 features
            top_n = min(20, len(feature_names))
            plt.bar(range(top_n), importances[indices[:top_n]], align='center')
            plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
            plt.title(f'Top {top_n} Feature Importances - {name}')
            plt.tight_layout()
            plt.savefig(f'output/models/{name}_feature_importance.png')
            plt.close()
    
    # Save results
    with open('models/model_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Determine best model based on F1 score
    best_model_name = max(results.items(), key=lambda x: x[1]['f1'])[0]
    print(f"\nBest model: {best_model_name}")
    print(f"F1 Score: {results[best_model_name]['f1']:.4f}")
    
    # Compare models
    plt.figure(figsize=(12, 8))
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    values = {model: [results[model][metric] for metric in metrics] for model in models}
    
    x = np.arange(len(metrics))
    width = 0.2
    
    for i, (model, vals) in enumerate(values.items()):
        plt.bar(x + i*width - 0.3, vals, width, label=model)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig('output/models/model_comparison.png')
    plt.close()
    
    print("\nModel training and evaluation completed.")
    return best_model_name, results

if __name__ == "__main__":
    train_evaluate_models()