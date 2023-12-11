import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import ks_2samp
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# Load dataset
df = pd.read_csv('creditcard.csv')

# Checking the first few rows
print(df.head())
df.isnull().sum().max()


#  Bar plot
fig, ax = plt.subplots()
sns.countplot(x='Class', data=df, ax=ax)
ax.set_title('Class Distribution')

# Add labels
total = len(df)
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}\n{height/total:.2%}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom')

# fitting labels
ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

plt.show()
fig.savefig('class_distribution_bar_plot.png')  # Save the figure



# histogram
fig, ax1 = plt.subplots(figsize=(12, 6))

num_bins = 40

# Histograma para a classe "Normal"
ax1.hist(df.Time[df.Class == 0], bins=num_bins, color='blue', alpha=0.5, label='Normal')

ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Transactions - Normal ')
ax1.set_title('Time Histogram for Normal X Fraud Transactions')

# Adiciona um segundo eixo y para a escala de "Fraude"
ax2 = ax1.twinx()
ax2.hist(df.Time[df.Class == 1], bins=num_bins, color='red', alpha=0.5, label='Fraud')

ax2.set_ylabel('Transactions - Fraud')

# Configuração da legenda
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper right')


plt.show()
fig.savefig('transaction_time_histogram.png')  # Save the figure



# boxplots


fig, ax = plt.subplots(figsize=(3,5), sharex=True)

sns.boxplot(x=df.Class, y=df.Amount, showmeans=True, ax=ax)
plt.ylim((-20, 400))
plt.xticks([0, 1], ['Normal', 'Fraud'])

plt.tight_layout()
plt.show()
fig.savefig('transaction_amount_boxplot.png')  # Save the figure



# Feature Selection techniques


# dataframe with independent variables
X = df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'
       , 'Amount'
        ]]

#intercept constant
X_int = sm.add_constant(X)

# LR model
model = sm.Logit(df['Class'], X_int)
result = model.fit()

# List os variables that didnt pass the tests
list_variables_to_drop = []




# Heteroscedasticity


X_temp = X_int.copy()
X_temp['const'] = 1

# DataFrame to input Breusch-Pagan tests
bp_results = pd.DataFrame(columns=['Feature', 'Breusch-Pagan', 'P-value'])


for variavel in X_int.columns:
    bp_test = het_breuschpagan(df['Class'], X_temp[[variavel, 'const']])
    bp_results = bp_results._append(pd.DataFrame({'Feature': [variavel], 'Breusch-Pagan': [bp_test[0]], 'P-value': [bp_test[1]]}), ignore_index=True)


pd.set_option('display.float_format', '{:.2f}'.format)

#putting variables in a list
for i in bp_results[bp_results['P-value'] > 0.05]['Feature']:
  if i not in list_variables_to_drop:
    list_variables_to_drop.append(i)

bp_results[bp_results['P-value'] > 0.05]


# Autocorrelation of the residuals

# Durbin-Watson test
from statsmodels.stats.stattools import durbin_watson
residuals = result.resid_response.copy()
dw = durbin_watson(residuals)
print('Durbin-Watson test = ', dw)


# Correlation between the independent variables

# VIF
vif = pd.DataFrame()
vif["Variável"] = X_int.columns
vif["VIF"] = [variance_inflation_factor(X_int.values, i) for i in range(X_int.shape[1])]

vif[vif['VIF']>5]
#Correlation matrix
corr = X.corr()

# Show only corr up to 0.4
mask = corr.abs() <= 0.4
corr_masked = corr.mask(mask)
corr_masked = corr_masked.applymap(lambda x: x*100 if abs(x) > 0.35 else np.nan)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_masked, xticklabels=corr.columns, yticklabels=corr.columns,
            linewidths=0.1, cmap="coolwarm", ax=ax, annot=True, fmt='.0f', annot_kws={"size": 8})
ax.set_title('Correlation Matrix')


plt.show()
fig.savefig('correlation_matrix.png')  # Save the figure

list_variables_to_drop += ['V2']
     
# Kolmogorov-Smirnov test
# KS test
ks_results = pd.DataFrame(columns=['Feature', 'D_test', 'P-value'])

for col in X.columns:
    label_1 = df[df['Class'] == 0][col]
    label_2 = df[df['Class'] == 1][col]
    statistic, p_value = ks_2samp(label_1, label_2)
    ks_results = ks_results._append({'Feature': col, 'D_test': statistic, 'P-value': p_value}, ignore_index=True)

# showing results
ks_results = pd.DataFrame(ks_results)
for i in ks_results[ks_results['D_test'] <= 0.11]['Feature']:
  if i not in list_variables_to_drop:
    list_variables_to_drop.append(i)

ks_results[ks_results['D_test'] <= 0.11]

list_variables_to_drop


column_names = df.drop(['Class', 'Amount', 'Time'], axis=1).columns
num_plots = len(column_names)
df_class_0 = df[df.Class == 0]
df_class_1 = df[df.Class == 1]

fig, ax = plt.subplots(nrows=7, ncols=4, figsize=(18,18))
fig.subplots_adjust(hspace=1, wspace=1)

idx = 0
for col in column_names:
    idx += 1
    plt.subplot(7, 4, idx)
    sns.kdeplot(df_class_0[col], label="Class 0", fill=True)
    sns.kdeplot(df_class_1[col], label="Class 1", fill=True)
    plt.title(col, fontsize=10)
plt.tight_layout()
# plt.savefig()


# Standardize Time and Amount
# Standardize Time and Amount
df_clean = df.copy()
for i in list_variables_to_drop:
  if i in df_clean.columns:
    df_clean.drop(i,axis=1, inplace=True)
     

std_scaler = StandardScaler()
df_clean['std_amount'] = std_scaler.fit_transform(df_clean['Amount'].values.reshape(-1, 1))
df_clean['std_time'] = std_scaler.fit_transform(df_clean['Time'].values.reshape(-1, 1))
df_clean.drop(['Time', 'Amount'], axis=1, inplace=True)
df_clean.head()


# Train and Test split

X = df_clean.drop('Class', axis=1)
y = df_clean['Class']
#SMOTE
smote = SMOTE(sampling_strategy=1, k_neighbors=5, random_state=7)

#RandomUnderSampling
rus = RandomUnderSampler()

#Class weight
count_class_1 = y.value_counts()[0]
count_class_2 = y.value_counts()[1]
ratio = count_class_1/count_class_2
class_weight = {1:ratio, 0:1}

#ADASYN
adasyn = ADASYN(sampling_strategy=1, n_neighbors=5, random_state=7)

# Applying the model
accuracies = []
precisions = []
recalls = []
f1_scores = []
roc_aucs = []
fpr_list = []
tpr_list = []
lists_metrics = [accuracies,precisions,recalls,f1_scores,roc_aucs,fpr_list,tpr_list]

def model_lr(X_train, y_train, X_test, y_test, class_weight_t, lists, random_state=None):
  np.random.seed(2)
  lr = LogisticRegression(class_weight=class_weight_t, random_state=random_state)
  lr.fit(X_train, y_train)
  y_pred = lr.predict(X_test)
  y_pred_proba = lr.predict_proba(X_test)[:, 1]
  fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
  lists[5].append(fpr)
  lists[6].append(tpr)
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  f1 = f1_score(y_test, y_pred)
  roc_auc = roc_auc_score(y_test, y_pred)
  lists[0].append(accuracy)
  lists[1].append(precision)
  lists[2].append(recall)
  lists[3].append(f1)
  lists[4].append(roc_auc)
  return lists
     

#SMOTE
X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X, y, stratify=y, shuffle=True, random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_smote, y_train_smote)
lists_metrics_smote = model_lr(X_train_smote, y_train_smote, X_test_smote, y_test_smote, None, lists_metrics, random_state=42)
     

#RandomUnderSampler
X_train_rus, X_test_rus,  y_train_rus, y_test_rus = train_test_split(X, y, stratify=y, shuffle=True, random_state=123)
X_train_rus, y_train_rus = rus.fit_resample(X_train_rus, y_train_rus)
lists_metrics_rus = model_lr(X_train_rus, y_train_rus, X_test_rus, y_test_rus, None, lists_metrics, random_state=123)
     

#ADASYN
X_train_adasyn, X_test_adasyn, y_train_adasyn, y_test_adasyn = train_test_split(X, y, stratify=y, shuffle=True, random_state=456)
X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train_adasyn, y_train_adasyn)
lists_metrics_adasyn = model_lr(X_train_adasyn, y_train_adasyn, X_test_adasyn, y_test_adasyn, None, lists_metrics, random_state=456)
     

#Class Weight
X_train_cw, X_test_cw, y_train_cw, y_test_cw = train_test_split(X, y, stratify=y, shuffle=True, random_state=987)
lists_metrics_cw = model_lr(X_train_cw, y_train_cw, X_test_cw, y_test_cw, class_weight, lists_metrics, random_state=987)
     

#no balancing
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, random_state=1345)
lists_metrics = model_lr(X_train, y_train, X_test, y_test, None, lists_metrics, random_state=1345)


# Models evaluation
models = ['SMOTE','Random Under-Sampling', 'ADASYN', 'Weighted Classes', 'No Balancing']
colors = ['blue', 'green', 'orange', 'red', 'yellow']
     

def bar_plot(evaluation_m, ylabel,title):
  plt.figure(figsize=(9, 2))
  plt.bar(models, evaluation_m, color=colors)
  plt.xlabel('Models')
  plt.ylabel(ylabel)
  plt.title(title, pad=20)
  plt.ylim([0, 1.1])

  for i, acc in enumerate(evaluation_m):
      plt.text(i, acc, f'{acc:.2%}', ha='center', va='bottom')

     

bar_plot(lists_metrics[0],'Accuracy','Models Accuracys')
plt.savefig('accuracy_bar_plot.png')  # Save the figure
plt.close()

bar_plot(lists_metrics[1],'Precision','Models Precision')
plt.savefig('precision_bar_plot.png')  # Save the figure
plt.close() 
bar_plot(lists_metrics[2],'Recall','Models recall')
plt.savefig('recall_bar_plot.png')  # Save the figure
plt.close()
bar_plot(lists_metrics[3],'F1 scores','Models F1 score')
plt.savefig('f1_score_bar_plot.png')  # Save the figure
plt.close()


# Plot ROC-curve
plt.figure(figsize=(8, 6))
for fpr, tpr, model in zip(lists_metrics[5], lists_metrics[6], [smote, rus, adasyn, class_weight, None ]):
    plt.plot(fpr, tpr, label='Model: {}'.format(model))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Each Model')
plt.legend()
plt.show()
fig.savefig('ROCcurve.png') 


data = {
    'Model': models,
    'Accuracy': lists_metrics[0],
    'Precision': lists_metrics[1],
    'Recall': lists_metrics[2],
    'F1 Score': lists_metrics[3],
    'ROC AUC': lists_metrics[4]
}

metrics = pd.DataFrame(data)
print(metrics)


corr_matrix = df.corr()

# Plotting the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()