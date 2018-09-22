from tkinter import filedialog
from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
##import subprocess
#from contextlib import redirect_stdout
 
window = Tk()
#p=subprocess.Popen('./script',stdout=subprocess.PIPE,stderr=subprocess.PIPE)
#output,error=p.communicate()
window.title("Stock Prediction")
window.filename=''
def open_file():
	window.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("CSV files","*.csv"),("all files","*.*")))
	t1.delete('1.0','end-1c')
	t1.insert(END,window.filename)

def training():
	t3.delete('1.0','end-1c')
	t3.insert("1.0",'TRAINING')
	try:
		training_set=pd.read_csv(t1.get('1.0','end-1c'))
	except:
		t1.delete('1.0','end-1c')
		t1.insert(END,'Please Enter A Valid CSV Path')
		t3.delete('1.0','end-1c')
		return
	try:
		column=int(t2.get('1.0','end-1c'))
	except:
		t2.delete('1.0','end-1c')
		t2.insert(END,"Please Enter The Column Number")
		t3.delete('1.0','end-1c')
		return
	training_set=training_set.iloc[:,(column-1):column].values
	sc=MinMaxScaler()
	training_set=sc.fit_transform(training_set)
	X_train = training_set[0:-1]
	Y_train = training_set[1:]
	X_train = np.reshape(X_train,(X_train.size,1,1))
	regressor=Sequential()
	regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1)))
	regressor.add(Dense(units=1))
	regressor.compile(optimizer='rmsprop',loss='mean_squared_error')
	regressor.fit(x=X_train,y=Y_train,batch_size=32,epochs=200)
	t3.delete('1.0','end-1c')
	t3.insert("1.0",'TRAINED')
	inputs=Y_train[-1:]
	inputs=np.reshape(inputs,(1,1,1))
	t4.insert(END,str(sc.inverse_transform(regressor.predict(inputs))[0,0]))
	
def reset():
	t1.delete("1.0","end-1c")
	t2.delete("1.0","end-1c")
	t3.delete("1.0","end-1c")
	t4.delete("1.0","end-1c")
	t3.insert(END,"UNTRAINED")
	try:
		del training_set
		del column
		del sc
		del X_train
		del Y_train
		del regressor
		del inputs
	except:
			{}

l1=Label(window,text="Enter the Path to Stock Data CSV (Mandatory)")
l1.grid(row=0,column=0,padx=5,pady=5)
#t2=Text(window)
#t2.grid(row=7,column=0,padx=10,pady=10)
#redirect_stdout(t2)
t1=Text(window,height=2, width=45)
t1.grid(row=1,column=0,padx=10,pady=10)
b1=Button(window,text="Browse",height=2,width=10,command=open_file)
b1.grid(row=1,column=2,padx=5,pady=5)

l2=Label(window,text="Enter the Column Number with the Open Stock Price (Mandatory)")
l2.grid(row=2,column=0,padx=5,pady=5)
t2=Text(window,height=2, width=30)
t2.grid(row=3,column=0,padx=10,pady=10)
b2=Button(window,text="Start Training",height=2,width=20,command=training)
b2.grid(row=3, column=2,padx=5,pady=5)

l3=Label(window,text="Training Status (Training takes 90 to 120 seconds)")
l3.grid(row=4,column=0,padx=5,pady=5)
t3=Text(window,height=2, width=40)
t3.grid(row=5,column=0,padx=10,pady=10)
t3.insert(END,'UNTRAINED')
b2=Button(window,text="Reset",height=2,width=10,command=reset)
b2.grid(row=5, column=2,padx=5,pady=5)

l4=Label(window,text="Next Day Prediction Price")
l4.grid(row=6,column=0,padx=5,pady=5)
t4=Text(window,height=2, width=40)
t4.grid(row=7,column=0,padx=10,pady=10)



window.mainloop()

