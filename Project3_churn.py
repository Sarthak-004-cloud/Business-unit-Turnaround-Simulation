
import pandas as pd

df = pd.read_csv(r"C:\Users\thesa\Downloads\Project3_SaaS_Turnaround\B2B_SaaS_Turnaround_Churn.csv")

print("Shape:", df.shape)
df.head()


# Churn Distribution Overview

import seaborn as sns
import matplotlib.pyplot as plt

# Churn count plot
sns.countplot(x='Churn_Risk', data=df)
plt.title("Churn Risk Distribution (0 = Stable, 1 = At Risk)")
plt.xlabel("Churn Risk")
plt.ylabel("Number of Clients")
plt.show()


# Churn vs Satisfaction & Usage

# Boxplot: Satisfaction by churn
sns.boxplot(x='Churn_Risk', y='Satisfaction_Score', data=df)
plt.title("Satisfaction Score by Churn Risk")
plt.show()

# Boxplot: Usage by churn
sns.boxplot(x='Churn_Risk', y='Monthly_Usage_Hours', data=df)
plt.title("Monthly Usage Hours by Churn Risk")
plt.show()


# Segment by Profit vs Cost

# Scatterplot of Profit vs Cost
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Effective_Customer_Cost', y='Net_Profit', hue='Churn_Risk', data=df)
plt.title("Net Profit vs Cost to Serve (Colored by Churn Risk)")
plt.xlabel("Monthly Cost to Serve (₹)")
plt.ylabel("Net Profit (₹)")
plt.axhline(0, color='red', linestyle='--')
plt.show()


#preparing the data to showcase whether the clients are at risk of churning or not
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Select features we think impact churn
features = [
    'Monthly_Usage_Hours',
    'Support_Tickets_Raised',
    'Onboarding_Days',
    'Invoice_Amount',
    'Effective_Customer_Cost',
    'Contract_Length_Months',
    'Discount_Offered_%',
    'Client_Tenure_Months',
    'Renewals',
    'Satisfaction_Score'
]

# Target variable
X = df[features]
y = df['Churn_Risk']

# Split into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Now training the logistics regression model
# Create the model
model = LogisticRegression(max_iter=1000)

# Train it
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)


# TO AVOID THE ABOVE WARNING AND ALLOW CONVERGENCE

from sklearn.preprocessing import StandardScaler

# Scale features to help convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use scaled data for split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Re-train model on scaled features
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)


# Check performance
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Select features to cluster on (don’t include Churn_Risk or Client_ID)
cluster_features = [
    'Monthly_Usage_Hours',
    'Support_Tickets_Raised',
    'Invoice_Amount',
    'Effective_Customer_Cost',
    'Net_Profit',
    'Satisfaction_Score'
]

X_cluster = df[cluster_features]

# Optional: scale for better clustering
scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)


wcss = []  # Within-cluster sum of squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_cluster_scaled)
    wcss.append(kmeans.inertia_)

# Plot elbow
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal Cluster Count')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()



# Fit KMeans with chosen cluster count
kmeans = KMeans(n_clusters=3, random_state=42)
df['Segment_Label'] = kmeans.fit_predict(X_cluster_scaled)


# to view segment labels

df[['Client_ID', 'Segment_Label', 'Invoice_Amount', 'Net_Profit', 'Satisfaction_Score']].head(10)


df['Segment_Label'].value_counts()


# segment wise averages ( what makes them unique)
df.groupby('Segment_Label')[cluster_features].mean().round(2)


# Reduce to 2D for visualization
pca = PCA(n_components=2)
reduced = pca.fit_transform(X_cluster_scaled)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(reduced[:, 0], reduced[:, 1], c=df['Segment_Label'], cmap='Set2', s=50)
plt.title("Client Segments (via PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()

# Mapping personas to Segment_Label
persona_map = {
    0: 'Costly Revenue Generators',
    1: 'Low-Touch Profitable Clients',
    2: 'At-Risk Drainers'
}

# Apply
df['Segment_Persona'] = df['Segment_Label'].map(persona_map)


# Make a copy of the data to simulate changes
df_sim = df.copy()


# Remove clients who are draining value
df_sim = df_sim[df_sim['Segment_Persona'] != 'At-Risk Drainers']

# Identify them
mask_reprice = df_sim['Segment_Persona'] == 'Costly Revenue Generators'

# Reduce customer cost (simulate pricing fix)
df_sim.loc[mask_reprice, 'Effective_Customer_Cost'] *= 0.85

# Boost profit accordingly
df_sim.loc[mask_reprice, 'Net_Profit'] = df_sim['Invoice_Amount'] - df_sim['Effective_Customer_Cost']


#cutting on-boarding costs
# Assume ₹1500 cost saved per onboarding
df_sim['Net_Profit'] += 1500


# Compare Before vs After: Profit Impact


import matplotlib.pyplot as plt

# Total profits before and after
before = df['Net_Profit'].sum()
after = df_sim['Net_Profit'].sum()

# Plot
plt.bar(['Original', 'After Strategy'], [before, after], color=['gray', 'green'])
plt.title("Total Net Profit: Before vs After Strategy")
plt.ylabel("₹ Total Net Profit")
plt.show()

# Print change
print(f"Profit increased by ₹{after - before:,.0f}")


# NOW REVISED STRATEGY TO SHOWCASE A NEEDED POSITIVE IMPACT ( although sometimes it can backfire as we have seen )


# Reset simulation base
df_sim = df.copy()


# Drop non-recoverable drainers
drop_mask = (df_sim['Segment_Persona'] == 'At-Risk Drainers') & (df_sim['Net_Profit'] < 0)
df_sim = df_sim[~drop_mask]



#Drop any client from any segment if Net_Profit < 0
df_sim = df_sim[df_sim['Net_Profit'] >= 0]

#Reprice Costly Revenue Generators
mask_reprice = df_sim['Segment_Persona'] == 'Costly Revenue Generators'
df_sim.loc[mask_reprice, 'Effective_Customer_Cost'] *= 0.85
df_sim.loc[mask_reprice, 'Net_Profit'] = df_sim['Invoice_Amount'] - df_sim['Effective_Customer_Cost']

#Add automation-based onboarding cost savings
df_sim['Net_Profit'] += 1500


# FINAL GRAPH revised ( before vs after )


before = df['Net_Profit'].sum()
after = df_sim['Net_Profit'].sum()

plt.bar(['Original', 'After Refined Strategy'], [before, after], color=['gray', 'green'])
plt.title("Net Profit Impact After Smarter Turnaround Strategy")
plt.ylabel("₹ Total Net Profit")
plt.grid(True)
plt.show()

print(f" Net Profit changed by ₹{after - before:,.0f}")






