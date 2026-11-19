
from pathlib import Path
from collections import Counter

# --------------------------------------
# 1. Load the dataset
# --------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "all_sentiment_shuffled.txt"

def read_documents(doc_file):
    docs = []
    labels = []

    with open(doc_file, encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()

            # Skip topic and document ID
            docs.append(words[3:])

            # Keep the sentiment label
            labels.append(words[1])

    return docs, labels


all_docs, all_labels = read_documents(DATA_PATH)

print("Total documents:", len(all_docs))
print("Sentiment distribution:", Counter(all_labels))

# --------------------------------------
# 2. Split into training and evaluation
# --------------------------------------

split_point = int(0.80 * len(all_docs))

train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]

eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

print("\nTraining documents:", len(train_docs))
print("Evaluation documents:", len(eval_docs))

print("Training labels:", Counter(train_labels))
print("Evaluation labels:", Counter(eval_labels))

# --------------------------------------
# 3. Inspect an example
# --------------------------------------

print("\nFirst training document:")
print("Label:", train_labels[0])
print("First 20 words:", train_docs[0][:20])


# --------------------------------------
# 4. Train Multinomial Naive Bayes
# --------------------------------------

def train_nb(documents, labels, alpha=1.0):

    # Count the number of documents in each class
    class_counts = Counter(labels)

    # Count word occurrences separately for each class
    word_counts = {
        label: Counter() for label in class_counts
    }

    for doc, label in zip(documents, labels):
        word_counts[label].update(doc)

    # Vocabulary contains all distinct training words
    vocabulary = set()

    for counts in word_counts.values():
        vocabulary.update(counts.keys())

    # Total number of word occurrences in each class
    total_words = {
        label: sum(word_counts[label].values())
        for label in class_counts
    }

    # Class priors: P(class)
    priors = {
        label: class_counts[label] / len(documents)
        for label in class_counts
    }

    # Store everything required for classification
    model = {
        "priors": priors,
        "word_counts": word_counts,
        "total_words": total_words,
        "vocabulary": vocabulary,
        "alpha": alpha
    }

    return model


# --------------------------------------
# 5. Train and inspect our model
# --------------------------------------

model = train_nb(train_docs, train_labels)

print("\nQuestion: Naive Bayes Training")

print("Vocabulary size:", len(model["vocabulary"]))

for label in ["pos", "neg"]:
    print(f"\nClass: {label}")
    print(f"Prior probability: {model['priors'][label]:.6f}")
    print(f"Total word occurrences: {model['total_words'][label]}")
    print(f"Count of 'great': {model['word_counts'][label]['great']}")
    print(f"Count of 'bad': {model['word_counts'][label]['bad']}")

# --------------------------------------
# 6. Score a document using log probabilities
# --------------------------------------

import math

def score_doc_label(document, label, model):

    # Get parameters learned during training
    prior = model["priors"][label]
    counts = model["word_counts"][label]
    total = model["total_words"][label]
    alpha = model["alpha"]

    # Reserve one vocabulary entry for unknown words
    vocab_size = len(model["vocabulary"]) + 1

    denominator = total + alpha * vocab_size

    # Start with the log prior
    log_score = math.log(prior)

    # Add each word's log likelihood
    for word in document:

        word_count = counts[word]

        word_probability = (
            (word_count + alpha) / denominator
        )

        log_score += math.log(word_probability)

    return log_score


# --------------------------------------
# 7. Sanity checks
# --------------------------------------

print("\nQuestion: Naive Bayes Log Scores")

test_documents = [
    ["great"],
    ["bad"],
    ["a", "top-quality", "performance"]
]

for doc in test_documents:

    pos_score = score_doc_label(
        doc, "pos", model
    )

    neg_score = score_doc_label(
        doc, "neg", model
    )

    print(f"\nDocument: {doc}")
    print(f"Positive log score: {pos_score:.6f}")
    print(f"Negative log score: {neg_score:.6f}")

    # Higher log score indicates predicted label
    prediction = (
        "pos" if pos_score > neg_score else "neg"
    )

    print(f"Predicted sentiment: {prediction}")

    # For one-word examples, show normal probabilities
    if len(doc) == 1:
        print(
            "Positive joint probability:",
            f"{math.exp(pos_score):.8f}"
        )
        print(
            "Negative joint probability:",
            f"{math.exp(neg_score):.8f}"
        )

# --------------------------------------
# 8. Classify a single document
# --------------------------------------

def classify_nb(document, model):

    # Find the class with the highest log score
    best_label = max(
        model["priors"],
        key=lambda label: score_doc_label(
            document, label, model
        )
    )

    return best_label


# --------------------------------------
# 9. Test the classifier
# --------------------------------------

print("\nQuestion: Classify Single Documents")

for doc in test_documents:

    prediction = classify_nb(doc, model)

    print(f"Document: {doc}")
    print(f"Predicted sentiment: {prediction}\n")

# --------------------------------------
# 10. Classify multiple documents
# --------------------------------------

def classify_documents(docs, model):

    predictions = []

    for doc in docs:
        predicted_label = classify_nb(doc, model)
        predictions.append(predicted_label)

    return predictions


# --------------------------------------
# 11. Predict evaluation documents
# --------------------------------------

eval_predictions = classify_documents(
    eval_docs, model
)

print("\nQuestion: Classify Evaluation Set")

print("Number of predictions:", len(eval_predictions))

print("\nFirst 10 predictions:")
print(eval_predictions[:10])

print("\nFirst 10 true labels:")
print(eval_labels[:10])

# --------------------------------------
# 12. Calculate Classification Accuracy
# --------------------------------------

def accuracy(true_labels, guessed_labels):

    correct = 0

    for true, predicted in zip(true_labels, guessed_labels):
        if true == predicted:
            correct += 1

    return correct / len(true_labels)


# --------------------------------------
# 13. Evaluate Naive Bayes
# --------------------------------------

eval_accuracy = accuracy(
    eval_labels,
    eval_predictions
)

correct_predictions = sum(
    true == predicted
    for true, predicted in zip(
        eval_labels, eval_predictions
    )
)

incorrect_predictions = (
    len(eval_labels) - correct_predictions
)

print("\nQuestion: Naive Bayes Evaluation")
print("Total evaluation documents:", len(eval_labels))
print("Correct predictions:", correct_predictions)
print("Incorrect predictions:", incorrect_predictions)
print(f"Classification accuracy: {eval_accuracy:.6f}")
print(f"Classification accuracy (%): {eval_accuracy * 100:.2f}%")


# --------------------------------------
# 14. Error Analysis
# --------------------------------------

print("\nQuestion: Error Analysis")

# Find all incorrectly classified documents
misclassified_indices = []

for i in range(len(eval_docs)):
    if eval_labels[i] != eval_predictions[i]:
        misclassified_indices.append(i)

print("Total misclassified reviews:",
      len(misclassified_indices))

# Count the two types of mistakes
false_positives = sum(
    true == "neg" and predicted == "pos"
    for true, predicted in zip(
        eval_labels, eval_predictions
    )
)

false_negatives = sum(
    true == "pos" and predicted == "neg"
    for true, predicted in zip(
        eval_labels, eval_predictions
    )
)

print("Negative predicted as positive:",
      false_positives)
print("Positive predicted as negative:",
      false_negatives)

# --------------------------------------
# 15. Inspect selected incorrect reviews
# --------------------------------------

# Zero-based indices within the evaluation set
selected_indices = [545, 1092, 1813, 2069, 1558]

for i in selected_indices:

    doc = eval_docs[i]
    actual = eval_labels[i]
    predicted = eval_predictions[i]

    pos_score = score_doc_label(
        doc, "pos", model
    )

    neg_score = score_doc_label(
        doc, "neg", model
    )

    print(f"\n--- Evaluation Review {i} ---")
    print("Actual label:", actual)
    print("Predicted label:", predicted)
    print(f"Positive log score: {pos_score:.4f}")
    print(f"Negative log score: {neg_score:.4f}")

    # Show review text for manual examination
    print("Review:")
    print(" ".join(doc[:100]))



# --------------------------------------
# 16. Optional: Learning Curve
# --------------------------------------

import matplotlib.pyplot as plt

PLOTS_DIR = BASE_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

training_sizes = []
learning_accuracies = []

print("\nOptional Task: Learning Curve")

for percent in range(10, 101, 10):

    # Use 10%, 20%, ..., 100% of training data
    n = int(len(train_docs) * percent / 100)

    subset_docs = train_docs[:n]
    subset_labels = train_labels[:n]

    # Train a new Naive Bayes model
    subset_model = train_nb(
        subset_docs,
        subset_labels,
        alpha=1.0
    )

    # Always evaluate on the same test set
    predictions = classify_documents(
        eval_docs,
        subset_model
    )

    acc = accuracy(eval_labels, predictions)

    training_sizes.append(n)
    learning_accuracies.append(acc * 100)

    print(
        f"{percent:3d}% training ({n:4d} docs): "
        f"{acc * 100:.2f}% accuracy"
    )

# Plot the learning curve
plt.figure(figsize=(9, 6))

plt.plot(
    training_sizes,
    learning_accuracies,
    marker="o",
    linewidth=2
)

plt.xlabel("Number of Training Documents")
plt.ylabel("Evaluation Accuracy (%)")
plt.title("Naive Bayes Learning Curve")
plt.grid(alpha=0.3)
plt.ylim(70, 85)
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "naive_bayes_learning_curve.png",
    dpi=300
)

plt.show()

# --------------------------------------
# 17. Optional: Most Informative Words
# --------------------------------------

print("\nOptional Task: Informative Words")

# Use higher smoothing for stable word rankings
feature_alpha = 50.0

feature_model = train_nb(
    train_docs,
    train_labels,
    alpha=feature_alpha
)

pos_counts = feature_model["word_counts"]["pos"]
neg_counts = feature_model["word_counts"]["neg"]

vocab_size = len(feature_model["vocabulary"]) + 1

pos_denominator = (
    feature_model["total_words"]["pos"]
    + feature_alpha * vocab_size
)

neg_denominator = (
    feature_model["total_words"]["neg"]
    + feature_alpha * vocab_size
)

word_scores = []

for word in feature_model["vocabulary"]:

    # Exclude punctuation, short and rare tokens
    if not word.isalpha() or len(word) < 3:
        continue

    total_count = pos_counts[word] + neg_counts[word]

    if total_count < 40:
        continue

    pos_prob = (
        (pos_counts[word] + feature_alpha)
        / pos_denominator
    )

    neg_prob = (
        (neg_counts[word] + feature_alpha)
        / neg_denominator
    )

    log_ratio = math.log(pos_prob) - math.log(neg_prob)

    word_scores.append((word, log_ratio))

# Rank words using the log-likelihood ratio
top_positive = sorted(
    word_scores,
    key=lambda item: (-item[1], item[0])
)[:10]

top_negative = sorted(
    word_scores,
    key=lambda item: (item[1], item[0])
)[:10]

print("\nTop 10 Positive Words:")
for word, score in top_positive:
    print(f"{word:18s} {score:+.4f}")

print("\nTop 10 Negative Words:")
for word, score in top_negative:
    print(f"{word:18s} {score:+.4f}")

# --------------------------------------
# 18. Plot informative words
# --------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 6))

positive_words = top_positive[::-1]
negative_words = top_negative[::-1]

axes[0].barh(
    [word for word, score in positive_words],
    [score for word, score in positive_words],
    color="steelblue"
)

axes[0].set_title("Most Positive Words")
axes[0].set_xlabel("Positive Log-Likelihood Ratio")

axes[1].barh(
    [word for word, score in negative_words],
    [-score for word, score in negative_words],
    color="indianred"
)

axes[1].set_title("Most Negative Words")
axes[1].set_xlabel("Negative Log-Likelihood Ratio")

for ax in axes:
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)

fig.suptitle("Informative Words in Customer Reviews")
plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "naive_bayes_informative_words.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
