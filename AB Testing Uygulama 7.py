
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu,pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multicomp import MultiComparison

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


######################################################
# ANOVA (Analysis of Variance)
######################################################

# İkiden fazla grup ortalamasını karşılaştırmak için kullanılır.
# Burada sadece gruplar arasında farklılık var mıdır yok mudur şeklinde olacaktır.
# Yani sadece ikiden fazla grup olduğunda bu grupların ortalamaları arasında fark var mı yok mu sorusunu sorup devam ediyor olacağız.

df = sns.load_dataset("tips")
df.head()

df.groupby("day")["total_bill"].mean()

# Günler arasında ortalama açısından anlamlı bir fark mı ?

# 1. Hipotezleri kur

# HO: m1 = m2 = m3 = m4 Grup ortalamaları arasında fark yoktur.

# H1: Grup ortalamaları arasında fark vardır. Eşit değillerdir. En az birisi farklıdır.

# 2. Varsayım kontrolü

# Normallik varsayımı
# Varyans homojenliği varsayımı

# Varsayım sağlanıyorsa one way anova
# Varsayım sağlanmıyorsa kruskal

# H0: Normal dağılım varsayımı sağlanmaktadır.

for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, 'p-value: %.4f' % pvalue)
# p-value değerleri 0.05 den küçük olduğu için H0 hipotezi reddedilir.
# Normal dadğılım varsayımı sağlanmamaktadır.
# Bundan dolayı non-parametric test yapılması gerekir.

# H0: Varyans homojenliği varsayımı sağlanmaktadır.

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# p-value değerleri 0.05 den büyük olduğu için H0 hipotezi reddedilemez.
# Varyans homojenliği varsayımı sağlanmaktadır.


# 3. Hipotez testi ve p-value yorumu

# Hiç biri sağlamıyor.
df.groupby("day").agg({"total_bill": ["mean", "median"]})


# HO: Grup ortalamaları arasında istatistiki olarak anlamlı fark yoktur.

# parametrik anova testi:
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])
# p-value değerleri 0.05 den küçük olduğu için H0 hipotezi reddedilir.
# Grup ortalamaları arasında istatistiki olarak anlamlı fark vardır.

# Nonparametrik anova testi:
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])

# Fark hangi gruptan kaynaklanıyor?

comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())