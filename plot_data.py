import matplotlib.pyplot as plot
import pandas as pd

df1=pd.read_excel("./e1_result.xlsx")
df2=pd.read_excel("./norule_inter_partial.xlsx")
df3=pd.read_excel("./noreplace_related_onemin.xlsx")
df4=pd.read_excel("./replace_related_onemin.xlsx")
df5=pd.read_excel("./replace_inter.xlsx")
df6=pd.read_excel("./noreplace_inter.xlsx")
df=df1[['Epoch']].copy()
df['Experiment1']=df1['Training Loss']
df['Experiment2']=df2['Training Loss']
df['Experiment3']=df3['Training Loss']
df['Experiment4']=df4['Training Loss']
df['Experiment5']=df5['Training Loss']
df['Experiment6']=df6['Training Loss']
df=df.set_index('Epoch')
ax = df.plot.line()
ax.set_ylabel("MSE Loss")
plot.show()
ax.figure.savefig('trainloss.png')
df6=df6.set_index('Epoch')
ax = df6.plot.line()
ax.set_ylabel("MSE Loss")
plot.show()
ax.figure.savefig('df6.png')