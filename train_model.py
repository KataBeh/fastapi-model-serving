from sklearn.datasets import make_classification        # import för att skapa ett syntetiskt klassificeringsdataset
from sklearn.linear_model import LogisticRegression     # modellen som ska tränas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score              # behövs för att mäta hur stor andel av predictions som blir rätt
import joblib


# Skapar syntetisk kunddata:
X, y = make_classification(                             # x är features alltså mina imput, y är target alltså vad modellen ska förutsäga
    n_samples=1000,                                     # 1000 observationer
    n_features=4,                                       # varje observation har 4 imputvariabler
    n_informative=4,                                    # alla 4 innehåller info som hjälper modellen att skilja mellan klasserna
    n_redundant=0,                                      # skapar inga features som bara är kombinationer av andra features
    random_state=42                                     # samma slumpmässigt dataset
)


# Delar upp data:
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Skapar och tränar modellen:
model = LogisticRegression()

model.fit(X_train, y_train)                             # modellen tittar på x_train och jämför med y_train och försöker lära sig mönster i features

# Testar modellen:
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)          # jämför modelles predictioner med de riktiga svaren och räknar ut hur stor andel som blir rätt

print(f"Accuracy: {accuracy:.2f}")


# Sparar modellen:
joblib.dump(model, "model.joblib")                       # Ta den tränade Python-modellen och spara den som en fil

print("Model saved as model.joblib")