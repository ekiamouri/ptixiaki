import pandas as pd
import re
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


CATEGORY_KEYWORDS = {
    "positive_emotion": {
        "joy", "happy", "happiness", "love", "gratitude", "grateful", "hope", "hopeful",
        "excitement", "enthusiasm", "euphoria", "elation", "contentment", "optimism",
        "inspired", "inspiration", "pride", "confident", "success", "accomplishment",
        "blessed", "appreciation", "wonder", "amazement", "celebration", "kindness",
        "admiration", "affection", "acceptance", "adoration", "empowerment",
        "compassion", "tenderness", "fulfillment", "reverence", "zest", "playful",
        "freespirit", "resilience", "radiance", "whimsy", "harmony", "freedom"
    },

    "negative_emotion": {
        "sad", "sadness", "despair", "grief", "loneliness", "jealousy", "envy", "resentment",
        "frustration", "anxiety", "fear", "fearful", "intimidation", "helplessness",
        "disgust", "shame", "heartbreak", "betrayal", "sorrow", "regret", "overwhelmed",
        "devastated", "bitterness", "disappointment", "darkness", "isolation", "loss",
        "desperation", "desolation", "suffering", "melancholy", "numbness", "yearning",
        "apprehensive", "frustrated", "jealous", "envious", "boredom", "dismissive"
    },

    "emotion_reflection": {
        "serenity", "calm", "calmness", "mindfulness", "meditation", "reflection",
        "inner", "peace", "tranquility", "pensive", "contemplation", "self", "growth",
        "personal", "development", "healing", "cozy", "mood", "blues", "quiet", "stillness",
        "nostalgia", "curiosity", "confusion", "ambivalence", "determination",
        "acceptance", "thought", "journey", "memory", "dream", "future"
    },

    "lifestyle_travel": {
        "travel", "adventure", "road", "trip", "beach", "summer", "winter", "sunset", "nature",
        "park", "garden", "hiking", "ocean", "seaside", "mountain", "snow", "rainy", "days",
        "scenic", "views", "explore", "gems", "city", "food", "cooking", "dessert", "brunch",
        "baking", "coffee", "tea", "bread", "culinary", "fitness", "workout", "gym", "health",
        "wellness", "selfcare", "bikeride", "gardening", "coast", "trail", "homecooking"
    },

    "entertainment_technology": {
        "music", "concert", "movie", "film", "series", "theater", "standup", "comedy", "premiere",
        "night", "watch", "show", "tech", "ai", "vr", "coding", "virtual", "reality",
        "innovation", "gaming", "gamer", "online", "art", "painting", "gallery", "museum",
        "studio", "creative", "creativity", "fashion", "runway", "book", "reading", "author",
        "novel", "playlist", "artist", "cinema"
    },

    "social_education": {
        "family", "friends", "friendship", "community", "volunteer", "charity", "support", "connection",
        "career", "work", "productivity", "milestone", "project", "study", "learning", "language",
        "workshop", "blogging", "blog", "post", "newpost", "high", "school", "prom", "textbook",
        "cafeteria", "homework", "exams", "debater", "sciencefair", "teen", "crush", "selfie",
        "tiktok", "confessions", "senior", "memoir", "jazz", "wine", "aged", "lifetime",
        "mentor", "mentorship", "skill", "achievement", "class", "college"
    },

    "sports_politics_pets": {
        "sports", "marathon", "tennis", "basketball", "formula", "cycling", "hockey", "soccer",
        "cricket", "golf", "politics", "debate", "corruption", "injustice", "environment",
        "rights", "news", "argument", "pet", "puppy", "kittens", "adoption", "furry",
        "petadoption", "furryfriend", "petlove", "puppylove", "animal"
    }
}


CATEGORY_PRIORITY = [
    "positive_emotion",
    "negative_emotion",
    "emotion_reflection",
    "lifestyle_travel",
    "entertainment_technology",
    "social_education",
    "sports_politics_pets",
    "other"
]


def parse_hashtags(cell):
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s:
        return []
    
    tags = re.findall(r"#?([A-Za-z0-9_]+)", s)
    tags = [t.lower() for t in tags if t]
    return tags


def split_camel_case(s: str):
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", s)
    return [p.lower() for p in parts if p]


def tag_to_tokens(tag: str):
    tag_clean = tag.replace ("_", " ")
    tokens = []
    for piece in tag_clean.split():
        tokens.extend(split_camel_case(piece))
    return set(tokens)


def hashtag_frequency(df, top_n=20):
    all_tags = []
    for sublist in df["hashtags_list"]:
        for tag in sublist:
            all_tags.append(tag)
            
    counts = Counter(all_tags)
    
    print("\nΣύνολο μοναδικών hshtags: ", len(counts))
    print(f"\nTop {top_n} hashtags:")
    for tag, c in counts.most_common(top_n):
        print(f"{tag}: {c}")
        
    return counts


def categorize_tags(tag: str):
    tag = tag.lower()
    tokens = tag_to_tokens(tag)
    
    scores = {}
    for cat, words in CATEGORY_KEYWORDS.items():
        score = 0 
        
        for w in words:
            if w in tokens:
                score += 3
            
            if len(w) >= 4 and w in tag:
                score += 1
                
        scores[cat] = score
    
    best_cat = max(scores, key=scores.get)
    
    if scores[best_cat] == 0:
        return "other"
    return best_cat


def assign_category(tags):
    cats = {categorize_tags(t) for t in tags}
    if len(cats) > 1 and "other" in cats:
        cats.remove("other")
    return sorted(cats)


def category_frequency(df, top_n=20):
    all_cats = []
    for sublist in df["hashtag_categories"]:
        for cat in sublist:
            all_cats.append(cat)
            
    counts = Counter(all_cats)
    
    print("\nΣυχνότητα κατηγοριών: ")
    for cat, c in counts.most_common():
        print(f"{cat}: {c}")
        
    return counts


def get_primary_category(categories):
    for cat in CATEGORY_PRIORITY:
        if cat in categories:
            return cat
    return "other"


def primary_category_frequency(df):
    count_prim = Counter(df["primary_category"])
  
    print("\nΣυχνότητα primary category: ")
    for cat, c in count_prim.most_common():
        print(f"{cat}: {c}")
        
    return count_prim


def encoding_hashtags(df):
    #df["primary_category"] = df["hashtag_categories"].apply(get_primary_category)
    le = LabelEncoder()
    df['primary_category_encoded'] = le.fit_transform(df['primary_category'])
    print("\n")
    print(dict(zip(le.classes_, le.transform(le.classes_))))
    
    return df,le 


def load_data():
    df = pd.read_csv(r"C:\Users\Eleni\OneDrive\Υπολογιστής\sentimentdataset.csv", sep=';')
    
    print("Αρχικό dataset: ", df.shape)
    print("Στήλες: ", df.columns)
    print("Ελλιπείς τιμές: ", df.isnull().sum())
    duplicates = df.duplicated().sum()
    print(f"\nΠλήθος διπλότυπων γραμμών: {duplicates}")
    
    if "Hashtags" in df.columns:
        df["hashtags_list"] = df["Hashtags"].apply(parse_hashtags)
    else:
        raise KeyError("Δεν βρέθηκε στήλη 'Hashtags' στο αρχείο!")
        
    print("\nΔείγμα hashtags πριν και μετά:")
    print(df[["Hashtags", "hashtags_list"]].head(20))
    
    empty_tags = (df["hashtags_list"].str.len() == 0).sum()
    print(f'\nΓραμμές χωρίς hashtags: {empty_tags}')
    
    df["hashtag_categories"] = df["hashtags_list"].apply(assign_category)
    df["primary_category"] = df["hashtag_categories"].apply(get_primary_category)
    
    print("\nΔείγμα κατηγοριών: ")
    print(df[["hashtags_list", "hashtag_categories"]].head(20))
    
    print("\nΔείγμα primary category: ")
    print(df[["hashtags_list", "hashtag_categories", "primary_category"]].head(20))
    
    return df


def random_forest_classification(df, le):
    X = df[['Likes', 'Retweets']]
    y = df['primary_category_encoded']
    print("to x:", X)
    print("to y:" ,y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    
    print("y_test: ", y_test)
    print("\ny_pred: ", y_pred)
    
    y_test1 = y_test.values
    print("y_test1: ", y_test1)
    
    
    ##rf_model.accuracy
    ##print("\nrf_model.accuracy", rf_model.accuracy)
    
    
    accuracy = accuracy_score(y_test1, y_pred)
    print("\nAccuracy: ",accuracy)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    print("\nClassification Report: ", report)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    
    fig, ax = plt.subplots(figsize=(12,9))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
    plt.title("Confusion Matrix Random Forest")
    plt.tight_layout()
    plt.show()
    
    
    return rf_model


def knn_classification(df, le):
    X = df[['Likes', 'Retweets']]
    y = df['primary_category_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    
    y_pred = knn_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("\nknn accuracy: ", accuracy)
    report = classification_report(y_test, y_pred, target_names= le.classes_, zero_division=0)
    print("\nknn classification report: ", report)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    
    fig, ax = plt.subplots(figsize=(12,9))
    disp.plot(cmap=plt.cm.pink, xticks_rotation=45, ax=ax)
    plt.title("Confusion Matrix kNN")
    plt.tight_layout()
    plt.show()
    
    return knn_model


def svm_classification(df, le):
    X = df[['Likes', 'Retweets']]
    y = df['primary_category_encoded']
    
    X_train, X_test, y_train , y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
    
    svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    svm_model.fit(X_train, y_train)
    
    y_pred = svm_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("\nsvm accuracy: ", accuracy)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    print("\nsvm classification report: ", report)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    
    fig, ax = plt.subplots(figsize=(12,9))
    disp.plot(cmap=plt.cm.Greens, xticks_rotation=45, ax=ax)
    plt.title("Confusion Matrix SVM")
    plt.tight_layout()
    plt.show()
    
    return svm_model


def lstm_classification(df, le):
    X = df[['Likes', 'Retweets']].values
    y = df['primary_category_encoded'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_3d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_3d = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    lstm_model = Sequential([
        LSTM(64, input_shape=(1, 2), activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(le.classes_), activation='softmax')
        ])
    
    lstm_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    
    history = lstm_model.fit(X_train_3d, y_train, epochs=25, batch_size=16, verbose=1)
    
    y_pred_probs = lstm_model.predict(X_test_3d)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("\nLSTM Accuracy: ", accuracy)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    print("\nclassification report: ", report)
    
   
    
    
    
    plt.figure(figsize=(12, 5))
    

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='#2ecc71', marker='o')
    plt.title('LSTM Model Accuracy progression')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)


    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='#e74c3c', marker='o')
    plt.title('LSTM Model Loss reduction')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    disp.plot(cmap=plt.cm.Oranges, xticks_rotation=45, ax=ax)
    plt.title("Confusion Matrix LSTM")
    plt.tight_layout()
    plt.show()


    
    return lstm_model
    
def main():
    df = load_data()
    hashtag_counts = hashtag_frequency(df, top_n=20)
    category_counts = category_frequency(df)
    primary_counts = primary_category_frequency(df)
    
    df, le = encoding_hashtags(df)
    model = random_forest_classification(df, le)
    model2 = knn_classification(df, le)
    model3 = svm_classification(df, le)
    model4 = lstm_classification(df, le)
    
    return df, hashtag_counts, category_counts, primary_counts, model, model2, model3, model4

if __name__ == "__main__":
    main()