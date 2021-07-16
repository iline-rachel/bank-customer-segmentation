from flask import Flask,render_template,request
import pickle
import numpy as np

filename='customer-segmentation.pkl'
kmeans=pickle.load(open(filename,'rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def man():
    if request.method=='POST':
        Balance=float(request.form['BALANCE'])
        Oneoff_Purchases=float(request.form['ONEOFF_PURCHASES'])
        Installments_Purchases=float(request.form['INSTALLMENTS_PURCHASES'])
        Oneoff_Purchases_Frequency=int(request.form['ONEOFF_PURCHASES_FREQUENCY'])
        Purchases_Installments_Frequency=int(request.form['PURCHASES_INSTALLMENTS_FREQUENCY'])
        Cash_Advance_Frequency=int(request.form['CASH_ADVANCE_FREQUENCY'])
        Cash_Advance_TRX=int(request.form['CASH_ADVANCE_TRX'])
        Credit_Limit=float(request.form['CREDIT_LIMIT'])
        Payments=float(request.form['PAYMENTS'])
        Minimum_Payments=float(request.form['MINIMUM_PAYMENTS'])
        PRC_Full_Payment=int(request.form['PRC_FULL_PAYMENT'])
        Tenure=int(request.form['TENURE'])
        
        arr=np.array([[Balance,Oneoff_Purchases,Installments_Purchases,Oneoff_Purchases_Frequency,Purchases_Installments_Frequency,Cash_Advance_Frequency,Cash_Advance_TRX,Credit_Limit,Payments,Minimum_Payments,PRC_Full_Payment,Tenure]])
        pred=kmeans.predict(arr)
        
        return render_template('after.html',data=pred)


if __name__== "__main__":
    app.run(debug=True)