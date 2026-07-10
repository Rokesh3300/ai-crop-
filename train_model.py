import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Dataset-ai load panna (Kaggle-la download panna file-oda peyar correct-a irukkanum)
df = pd.read_csv('Crop_recommendation.csv')

# 2. Features (Input) and Label (Output)-ai separate panna
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# 3. Model-ai create panni train panna
model = RandomForestClassifier()
model.fit(X, y)

# 4. Idhu thaan mukkiyamanathu: Brain-ai .pkl file-ah save pannum
joblib.dump(model, 'crop_model.pkl')

print("Model success! 'crop_model.pkl' file ippo unga folder-la irukkum.")