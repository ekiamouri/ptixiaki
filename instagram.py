############## ΟΝΟΜΑΤΕΠΩΝΥΜΟ: ΚΙΑΜΟΥΡΗ ΕΛΕΝΗ, ΑΜ: 2122157 
############## DATASET: https://www.kaggle.com/datasets/kashishparmar02/social-media-sentiments-analysis-dataset?utm_source=chatgpt.com



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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, concatenate, Embedding, SpatialDropout1D, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


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
    
    """
    print("\nΣύνολο μοναδικών hshtags: ", len(counts))
    print(f"\nTop {top_n} hashtags:")
    for tag, c in counts.most_common(top_n):
        print(f"{tag}: {c}")
    """
    
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
    
    """
    print("\nΣυχνότητα κατηγοριών: ")
    for cat, c in counts.most_common():
        print(f"{cat}: {c}")
    """ 
    
    return counts


def get_primary_category(categories):
    for cat in CATEGORY_PRIORITY:
        if cat in categories:
            return cat
    return "other"


def primary_category_frequency(df):
    count_prim = Counter(df["primary_category"])
    
    """
    print("\nΣυχνότητα primary category: ")
    for cat, c in count_prim.most_common():
        print(f"{cat}: {c}")
    """
    
    return count_prim


def encoding_hashtags(df):
    #df["primary_category"] = df["hashtag_categories"].apply(get_primary_category)
    le = LabelEncoder()
    df['primary_category_encoded'] = le.fit_transform(df['primary_category'])
    """
    print("\n")
    print(dict(zip(le.classes_, le.transform(le.classes_))))
    """
    
    return df,le 


def load_data():
    df = pd.read_csv(r"C:\Users\Eleni\OneDrive\Υπολογιστής\sentimentdataset.csv", sep=';')
    
    if "Hashtags" in df.columns:
        df["hashtags_list"] = df["Hashtags"].apply(parse_hashtags)
    else:
        raise KeyError("Δεν βρέθηκε στήλη 'Hashtags' στο αρχείο!")
        
    df["hashtag_categories"] = df["hashtags_list"].apply(assign_category)
    df["primary_category"] = df["hashtag_categories"].apply(get_primary_category)
    
    df["hashtags_Count"] = df["hashtags_list"].apply(len)
    
    le_platform = LabelEncoder()
    df["Platform_encoded"] = le_platform.fit_transform(df["Platform"])
    
    le_sentiment = LabelEncoder()
    df["Sentiment_encoded"] = le_sentiment.fit_transform(df['Sentiment'])
    
    df["metadata_text"] = (
        df["Platform"].astype(str) + " " +
        df["Hashtags"].astype(str) + " " +
        df["Sentiment"].astype(str)
        )
    
    return df


def random_forest_classification(df, le):
    X = df[['Likes', 'Retweets']]
    y = df['primary_category_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    
    y_test1 = y_test.values    
    
    accuracy = accuracy_score(y_test1, y_pred)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print("\t-Random Forest Accuracy: ",accuracy)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    print("\n\t-Random Forest Classification Report: ", report)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    
    fig, ax = plt.subplots(figsize=(12,9))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
    plt.title("Confusion Matrix Random Forest")
    plt.tight_layout()
    plt.show()
    
    return rf_model, accuracy


def knn_classification(df, le):
    X = df[['Likes', 'Retweets']]
    y = df['primary_category_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    
    y_pred = knn_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print("\t-KNN Accuracy: ", accuracy)
    report = classification_report(y_test, y_pred, target_names= le.classes_, zero_division=0)
    print("\n\t-KNN Classification Report: ", report)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    
    fig, ax = plt.subplots(figsize=(12,9))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
    plt.title("Confusion Matrix kNN")
    plt.tight_layout()
    plt.show()
    
    return knn_model, accuracy


def svm_classification(df, le):
    X = df[['Likes', 'Retweets']]
    y = df['primary_category_encoded']
    
    X_train, X_test, y_train , y_test = train_test_split(X, y ,test_size=0.2, random_state=42)
    
    svm_model = SVC(kernel='rbf', class_weight='balanced', random_state=42)
    svm_model.fit(X_train, y_train)
    
    y_pred = svm_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print("\t-SVM Accuracy: ", accuracy)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    print("\n\t-SVM Classification Report: ", report)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    
    fig, ax = plt.subplots(figsize=(12,9))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
    plt.title("Confusion Matrix SVM")
    plt.tight_layout()
    plt.show()
    
    return svm_model, accuracy


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
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print("\t-LSTM Accuracy: ", accuracy)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    print("\n\t-LSTM Classification Report: ", report)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")    
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='lightblue', marker='o')
    plt.title('LSTM Model Accuracy progression')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='purple', marker='o')
    plt.title('LSTM Model Loss reduction')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
    plt.title("Confusion Matrix LSTM")
    plt.tight_layout()
    plt.show()

    return lstm_model, accuracy


def bilstm_classification(df, le):
    X = df[['Likes', 'Retweets']].values
    y = df['primary_category_encoded'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_3d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_3d = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    bilstm_model = Sequential([
        Bidirectional(LSTM(64, input_shape=(1, 2), activation='relu')),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(le.classes_), activation='softmax')
        ])
    
    bilstm_model.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    
    history = bilstm_model.fit(X_train_3d, y_train, epochs=25, batch_size=16, verbose=1)
    
    y_pred_probs = bilstm_model.predict(X_test_3d)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print("\t-Bi-LSTM Accuracy: ", accuracy)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    print("\n\t-Bi-LSTM Classification Report: ", report)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='lightblue', marker='o')
    plt.title('Bi-LSTM Model Accuracy progression')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='purple', marker='o')
    plt.title('Bi-LSTM Model Loss reduction')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
    plt.title("Confusion Matrix Bi-LSTM")
    plt.tight_layout()
    plt.show()

    return bilstm_model, accuracy

 
def bilstm_classification2(df, le):
    X = df[['Likes', 'Retweets', 'hashtags_Count', 'Platform_encoded']].values
    y = df['primary_category_encoded'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_3d = X_train.reshape((X_train.shape[0], 1, 4))
    X_test_3d = X_test.reshape((X_test.shape[0], 1, 4))
    
    bilstm_model2 = Sequential([
        Input(shape=(1,4)),
        Bidirectional(LSTM(64, activation='relu')),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(le.classes_), activation='softmax')
        ])
    
    bilstm_model2.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
    
    history = bilstm_model2.fit(X_train_3d, y_train, epochs=30, batch_size=16, verbose=1)
    
    y_pred_probs = bilstm_model2.predict(X_test_3d)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print("\t-Bi-LSTM2 Accuracy: ", accuracy)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    print("\n\t-Bi-LSTM2 Classification Report: ", report)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='lightblue', marker='o')
    plt.title('Bi-LSTM2 Model Accuracy progression')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='purple', marker='o')
    plt.title('Bi-LSTM2 Model Loss reduction')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
    plt.title("Confusion Matrix Bi-LSTM2")
    plt.tight_layout()
    plt.show()

    return bilstm_model2, accuracy


def bilstm_hybrid(df, le):
    X1 = df[['Likes', 'Retweets']].values
    X2 = df[['hashtags_Count', 'Platform_encoded', 'Sentiment_encoded']].values
    y = df['primary_category_encoded'].values
    
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)
    
    X1_train_3d = X1_train.reshape((X1_train.shape[0], 1 ,2))
    X1_test_3d = X1_test.reshape((X1_test.shape[0], 1, 2))
    X2_train_3d = X2_train.reshape((X2_train.shape[0], 1, 3))
    X2_test_3d = X2_test.reshape((X2_test.shape[0], 1, 3))
    
    input1 = Input(shape=(1, 2))
    lstm1 = Bidirectional(LSTM(64, activation='relu'))(input1)
    drop1 = Dropout(0.2)(lstm1)
    
    input2 = Input(shape=(1, 3))
    lstm2 = Bidirectional(LSTM(64, activation='relu'))(input2)
    drop2 = Dropout(0.2)(lstm2)
    
    merged = concatenate([drop1, drop2])
    
    dense1 = Dense(32, activation='relu')(merged)
    output = Dense(len(le.classes_), activation='softmax')(dense1)
    
    model = Model(inputs=[input1, input2], outputs=output)
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit([X1_train_3d, X2_train_3d], y_train, epochs=30, batch_size=16, verbose=1)
    
    y_pred_probs = model.predict([X1_test_3d, X2_test_3d])
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print("\t-Bi-LSTM-hybrid Accuracy: ", accuracy)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    print("\n\t-Bi-LSTM-hybrid Classification Report: ", report)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='lightblue', marker='o')
    plt.title('Bi-LSTM-hybrid Model Accuracy progression')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='purple', marker='o')
    plt.title('Bi-LSTM-hybrid Model Loss reduction')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
    plt.title("Confusion Matrix Bi-LSTM-hybrid")
    plt.tight_layout()
    plt.show()
    
    return model, accuracy


def bilstm_numeric_text(df, le):
    X1 = df[['Likes', 'Retweets']].values
    X2 = df['metadata_text']
    y = df['primary_category_encoded'].values
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X2)
    sequences = tokenizer.texts_to_sequences(X2)
    X2_padded = pad_sequences(sequences, maxlen=15)
    
    vocab_size = len(tokenizer.word_index) + 1
    
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2_padded, y, test_size=0.2, random_state=42)
    
    X1_train_3d = X1_train.reshape((X1_train.shape[0], 1 , 2))
    X1_test_3d = X1_test.reshape((X1_test.shape[0], 1, 2))
    
    input_num = Input(shape=(1, 2))
    lstm_num = Bidirectional(LSTM(128))(input_num)
    drop_num = Dropout(0.3)(lstm_num)
    
    input_text = Input(shape=(15, ))
    embed_text = Embedding(input_dim=vocab_size, output_dim=128)(input_text)
    spatial_drop = SpatialDropout1D(0.2)(embed_text)
    
    lstm_text_1 = Bidirectional(LSTM(64, return_sequences=True))(spatial_drop)
    lstm_text_2 = Bidirectional(LSTM(64))(lstm_text_1)
    drop_text = Dropout(0.3)(lstm_text_2)
    
    merged = concatenate([drop_num, drop_text])
    
    dense = BatchNormalization()(merged)
    dense = Dense(128, activation='relu')(dense)
    dense = Dropout(0.3)(dense)
    
    output = Dense(len(le.classes_), activation='softmax')(dense)
    
    model = Model(inputs=[input_num, input_text], outputs=output)
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    history = model.fit(x=[X1_train_3d, X2_train], y=y_train, epochs=40, batch_size=32, verbose=1)
    
    y_pred_probs = model.predict([X1_test_3d, X2_test])
    y_pred = np.argmax(y_pred_probs, axis=1)   
    
    accuracy = accuracy_score(y_test, y_pred)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    print("\t-Bi-LSTM Numeric + Text Accuracy: ", accuracy)
    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    print("\n\t-Bi-LSTM Numeric + Text Classification Report: ", report)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='lightblue', marker='o')
    plt.title('Bi-LSTM Numeric + Text Model Accuracy progression')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='purple', marker='o')
    plt.title('Bi-LSTM Numeric + Text Model Loss reduction')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
    plt.title("Confusion Matrix Bi-LSTM Numeric + Text")
    plt.tight_layout()
    plt.show()
    
    return model, accuracy


def plot_model_accuracies(results):
    model_names = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(12, 7))

    bars = plt.bar(model_names, accuracies)

    plt.title("Comparison of Model Accuracy")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")

    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{accuracy * 100:.2f}%",
            ha="center",
            va="bottom"
        )

    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    
def main():
    df = load_data()
    hashtag_counts = hashtag_frequency(df, top_n=20)
    category_counts = category_frequency(df)
    primary_counts = primary_category_frequency(df)
    
    df, le = encoding_hashtags(df)
    model, rf_accuracy = random_forest_classification(df, le)
    model2, knn_accuracy = knn_classification(df, le)
    model3, svm_accuracy = svm_classification(df, le)
    model4, lstm_accuracy = lstm_classification(df, le)
    model5, bilstm_accuracy = bilstm_classification(df, le)
    model6, bilstm2_accuracy = bilstm_classification2(df, le)
    model7, hybrid_accuracy = bilstm_hybrid(df, le)
    model8, numeric_text_accuracy = bilstm_numeric_text(df, le)
    
    accuracy_results = {
        "Random Forest": rf_accuracy,
        "kNN": knn_accuracy,
        "SVM": svm_accuracy,
        "LSTM": lstm_accuracy,
        "Bi-LSTM": bilstm_accuracy,
        "Bi-LSTM2": bilstm2_accuracy,
        "Bi-LSTM Hybrid": hybrid_accuracy,
        "Bi-LSTM Numeric + Text": numeric_text_accuracy
        }

    plot_model_accuracies(accuracy_results)
    
    return df, hashtag_counts, category_counts, primary_counts, model, model2, model3, model4, model5, model6, model7, model8

    
    return df, hashtag_counts, category_counts, primary_counts, model, model2, model3, model4, model5, model6, model7, model8


if __name__ == "__main__":
    main()