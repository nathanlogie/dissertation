from sklearn.metrics import accuracy_score


def adaptive_training_loop(X, y, sensitive_attr, adaptive_model, num_epochs=10, tolerance=0.01):
    """
    Adaptive training loop to ensure bias score decreases over epochs.
    """
    current_bias_score = float('inf')  # Start with a very high bias score
    bias_scores = []
    accuracy_scores = []

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}")

        # Train the model
        adaptive_model.train(X, y, sensitive_attr)
        y_pred = adaptive_model.predict(X)

        # Evaluate bias score
        new_bias_score = adaptive_model.evaluate_bias(X, y, sensitive_attr)
        bias_scores.append(new_bias_score)

        print(f"Epoch {epoch}: Bias Score = {new_bias_score}")

        # Check if bias score is improving
        if new_bias_score >= current_bias_score - tolerance:
            print("Epoch {epoch}: Sensitive attribute masked.")
            adaptive_model.update_masking(X, sensitive_attr)  # Apply adaptive masking

        # Update the current bias score
        current_bias_score = new_bias_score

        # Evaluate accuracy or other metrics
        accuracy = accuracy_score(y, y_pred)
        accuracy_scores.append(accuracy)

    print("\nFinal Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Bias Scores: {bias_scores}")

    return bias_scores, accuracy_scores
