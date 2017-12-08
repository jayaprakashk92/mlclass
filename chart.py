import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="whitegrid")
#df = pd.DataFrame(sns.load_dataset("flights"))

df=sns.load_dataset("flights")

df=df[df.year > 1953]



print (df)


sns.boxplot(x="year", y="passengers",whis=1,data=df,palette="vlag")

sns.swarmplot(x="year", y="passengers",hue="month", data=df,size=6, linewidth=1)

plt.show()
