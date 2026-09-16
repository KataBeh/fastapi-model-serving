# from sklearn.datasets import make_classification        # import för att skapa ett syntetiskt klassificeringsdataset
from sklearn.linear_model import LogisticRegression       # modellen som ska tränas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score                # behövs för att mäta hur stor andel av predictions som blir rätt
import joblib
import numpy as np


# För reproducerbarhet:
rng = np.random.default_rng(42)

# Skapar syntetisk kunddata:
n_customers = 1000

monthly_spend = rng.integers(100, 1500, size=n_customers)   # kunder får en månadskostnad mellan 100 och 1500          
months_as_customer = rng.integers(1, 61, size=n_customers)
support_cases = rng.integers(0, 11, size=n_customers)       # mellan 0 till 10 supportärenden
usage_frequency = rng.integers(1, 31, size=n_customers)

# Kombinerar features till en X-matris:
X = np.column_stack([
    monthly_spend,
    months_as_customer,
    support_cases,
    usage_frequency
])


# Skapar ett enkelt syntetiskt churn-samband:
churn_score = (                                             # alltså ett påhittat mått på hur sannolikt det är att kunden lämnar
    0.003 * monthly_spend
    - 0.05 * months_as_customer
    + 0.5 * support_cases
    - 0.15 * usage_frequency
)


# Lägger till lite slump så att problemet inte blir helt perfekt:
noise = rng.normal(0, 1, size=n_customers)

churn_score = churn_score + noise


# Om churn_score är större än 1 klassas kunden som "leave":
y = (churn_score > 1).astype(int)

# Delar upp data:
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Skapar och tränar modellen:
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)                             # modellen tittar på x_train och jämför med y_train och försöker lära sig mönster i features

# Utvärderar modellen:
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)          # jämför modelles predictioner med de riktiga svaren och räknar ut hur stor andel som blir rätt

print(f"Accuracy: {accuracy:.2f}")


# Sparar modellen:
joblib.dump(model, "model.joblib")                       # Ta den tränade Python-modellen och spara den som en fil

print("Model saved as model.joblib")