import pandas as pd
import numpy as np
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima_model import ARMA,ARMAResults,ARIMA,ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from fbprophet import Prophet
import math

df = pd.read_csv('airline_passengers.csv',index_col='Month',parse_dates=True)
df.index.freq = 'MS'
df.head()

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true).reshape(len(y_true),1), np.array(y_pred).reshape(len(y_true),1)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def transform_series(df):
    col=df.columns[0]
    z=0
    for i in np.arange(len(df)-1,-1,-1):
        if df[col].iloc[i]==0:
            z=1
            break
    if z==0:
        return df
    return df[i+1:]

from numpy import percentile
def outlier(data):
    col=data.columns[0]
    q25, q75 = percentile(data, 25), percentile(data, 75)
#     print(q25,q75)
    iqr = q75 - q25
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    x=len(data[data[col]<lower])+len(data[data[col]>upper])
    print(x)
    for i in range(0,len(data)):
        if data[col][i]<lower or data[col][i]>upper:
#             print(i)
#             print(data[col][i-1])
            data[col][i]=data[col][i-1]
    return data

def exp(df,horizon):
    n_train = int(len(df)*0.7)
    n_records = len(df)
    MAPE_add=0
    MAPE_mul=0
    j=0
    import math
    col=df.columns[0]
    for i in range(n_train, n_records-horizon+1):
        try:
            train, test = df.iloc[0:i], df.iloc[i:i+horizon]
#             print('train=%d, test=%d' % (len(train), len(test)))
            fitted_model = ExponentialSmoothing(train[col],trend='add').fit()
            test_predictions= fitted_model.forecast(len(test))
            MAPE_add+=mean_absolute_percentage_error(test,test_predictions)
#             print(MAPE_add)
            fitted_model = ExponentialSmoothing(train[col],trend='mul').fit()
            test_predictions= fitted_model.forecast(len(test))
            MAPE_mul+=mean_absolute_percentage_error(test,test_predictions)
#             print(MAPE_mul)
            j+=1
        except:
            pass
    if (MAPE_add<MAPE_mul or math.isnan(MAPE_mul)):
        return {'trend':'add','MAPE':MAPE_add/j}
    return {'trend':'mul','MAPE':MAPE_mul/j}   

def exp_seasonal(df,horizon):
    col=df.columns[0]
    details={'trend':'','seasonal':'','MAPE':100}
    MAPE=10000
    n_train = int(len(df)*0.7)
    n_records = len(df)
    for i in ['add','mul']:
        for j in ['add','mul']:
            k=0
            mape=0
            for l in range(n_train, n_records-horizon+1):
                try:
                    train, test = df.iloc[0:l], df.iloc[l:l+horizon]
#                     print('train=%d, test=%d' % (len(train), len(test)))
                    fitted_model = ExponentialSmoothing(train[col],trend=i,seasonal=j,seasonal_periods=12).fit()
                    test_predictions= fitted_model.forecast(len(test))
                    x=mean_absolute_percentage_error(test,test_predictions)
                    if x>0:
                        mape+=x
                        k+=1
                except:
                    pass
            if k==0:
                continue
            mape=mape/k
            if mape<MAPE:
                MAPE=mape
                details['trend']=i
                details['seasonal']=j
                details['MAPE']=MAPE
    return details

from statsmodels.tsa.stattools import adfuller

def adf_test(series):
    result=1
    d=-1
    while(result>0.05):
        result = adfuller(series.dropna(),autolag='AIC')[1]
        d+=1
        series=series.diff()
    return d

def arima(df,horizon):
    col=df.columns[0]
    d= adf_test(df[col])
    d=min(d,2)
    AIC=ARIMA(df[col],order=(0,d,0)).fit().aic
#     print((0,d,0))
#     print(AIC)
    details={'order':(0,d,0),'AIC':AIC, 'MAPE':100}
    for p in range(0,6):
        for q in range(0,6):
            if (p==0 and q==0):
                continue
            try:
                results = ARIMA(df[col],order=(p,d,q)).fit()
                aic=results.aic
#                 print((p,d,q))
#                 print(aic)
                if aic<AIC:
                    AIC=aic
                    details['order']=(p,d,q)
                    details['AIC']=aic
            except:
                pass
    n_train = int(len(df)*0.7)
    n_records = len(df)
    k=0
    mape=0
    for l in range(n_train, n_records-horizon+1):
        try:
            train, test = df.iloc[0:l], df.iloc[l:l+horizon]
#           print('train=%d, test=%d' % (len(train), len(test)))
            model = ARIMA(train[col],order=details['order'])
#             model = ARIMA(train[col],order=order)
            results=model.fit()
            start=len(train)
            end=len(train)+len(test)-1
            test_predictions = results.predict(start=start, end=end, dynamic=False, typ='levels')
            x=mean_absolute_percentage_error(test,test_predictions)
            if x>0:
                mape+=x
                k+=1
        except:
            pass
    if k>0:
        mape=mape/k
        details['MAPE']=mape
    return details

def seasonal_arima(df,horizon):
    col=df.columns[0]
    AIC=SARIMAX(df[col],order=(0,0,0),seasonal_order=(0,0,0,12)).fit().aic
    details={'order':(0,0,0),'seasonal_order':(0,0,0,12),'AIC':AIC, 'MAPE':100}
    for d in range(0,2):
        for p in range (0,3):
            for q in range(0,3):
                for P in range(0,3):
                    for Q in range(0,3):
                        for D in range (0,2):
                            if P==0 and D==0 and Q==0 and p==0 and q==0 and d==0:
                                continue
                            try:
                                model = SARIMAX(df[col],order=(p,d,q),seasonal_order=(P,D,Q,12))
                                results = model.fit(maxiter=1000)
                                aic=results.aic
                                if aic<AIC:
                                    AIC=aic
                                    details['order']=(p,d,q)
                                    details['seasonal_order']=(P,D,Q,12)
                                    details['AIC']=aic
                            except:
                                pass
    n_train = int(len(df)*0.7)
    n_records = len(df)
    k=0
    mape=0
    for l in range(n_train, n_records-horizon+1):
        try:
            train, test = df.iloc[0:l], df.iloc[l:l+horizon]
#           print('train=%d, test=%d' % (len(train), len(test)))
            model = SARIMAX(train[col],order=details['order'],seasonal_order=details['seasonal_order'])
#             model = SARIMAX(train[col],order=order,seasonal_order=seasonal_order)
            results=model.fit(maxiter=1000)
            start=len(train)
            end=len(train)+len(test)-1
            test_predictions = results.predict(start=start, end=end, dynamic=False, typ='levels')
            x=mean_absolute_percentage_error(test,test_predictions)
            if x>0:
                print(x)
                mape+=x
                k+=1
        except:
            pass
    if k>0:
        mape=mape/k
        details['MAPE']=mape
    return details

def rnn(df,n_input,horizon,e):
    details ={'LSTM':100,'Input':n_input, 'MAPE':100}
    n_train = int(len(df)*0.7)
    if n_input+4>n_train:
        return details
    n_records = len(df)
    k=0
    mape=0
    for l in range(n_train, n_records-horizon+1):
        train, test = df.iloc[0:l], df.iloc[l:l+horizon]
        scaler = MinMaxScaler()
        scaler.fit(train)
        scaled_train = scaler.transform(train)
        n_features = 1
        generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)
        model = Sequential()
        model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit_generator(generator,epochs=e)
        forecast = []
        first_eval_batch = scaled_train[-n_input:]
        current_batch = first_eval_batch.reshape((1, n_input, n_features))
        for i in range(len(test)):
            current_pred = model.predict(current_batch)[0]
            forecast.append(current_pred) 
            current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
        forecast= scaler.inverse_transform(forecast)
        mape+=mean_absolute_percentage_error(test,forecast)
        k+=1
    details['MAPE']=mape/k
    return details 

def ann(df,n_input,horizon):
    t=int((n_input+1)/2)
    details ={'Dense':(t),'Input':n_input, 'MAPE':100}
    n_train = int(len(df)*0.7)
    if n_input+4>n_train:
        return details
    n_records = len(df)
    k=0
    mape=0
    for l in range(n_train, n_records-horizon+1):
        train, test = df.iloc[0:l], df.iloc[l:l+horizon]
        scaler = MinMaxScaler()
        scaler.fit(train)
        scaled_train = scaler.transform(train)
        X=[]
        y=[]
        j=0
        for i in range(n_input,len(scaled_train)):
            X.append(scaled_train[j:i].reshape(1,n_input))
            y.append(scaled_train[i])
            j+=1
        X=np.array(X).reshape(j,n_input)
        y=np.array(y).reshape(j,1)
        model = Sequential()
        model.add(Dense(int((n_input+1)/2), activation='relu', input_dim=n_input))
#         model.add(Dense(int((n_input+1)/2), activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X,y,epochs=300)
        forecast = []
        eval_batch = scaled_train[-n_input:].reshape(1,n_input)
        for i in range(len(test)):
            current_pred = model.predict(eval_batch)
            forecast.append(current_pred[0]) 
            eval_batch=np.append(eval_batch[:,1:],current_pred[0][0]).reshape(1,n_input)
        forecast= scaler.inverse_transform(forecast)
        mape+=mean_absolute_percentage_error(test,forecast)
        k+=1
    details['MAPE']=mape/k
    return details

def nnetar(df,p,P,horizon):
    n_input=p+P
    t=int((n_input+1)/2)
    details ={'Dense':4,'Input':n_input, 'MAPE':100}
    if (len(df)<12*P+10):
        return details
    n_train=12*P+5
    n_records = len(df)
    t=0
    mape=0
    for l in range(n_train, n_records-horizon+1):
        train, test = df.iloc[0:l], df.iloc[l:l+horizon]
        scaler = MinMaxScaler()
        scaler.fit(train)
        scaled_train = scaler.transform(train)
        X=[]
        y=[]
        j=0
        for i in range(max(12*P,p),len(scaled_train)):
            X.append(scaled_train[i-p:i].reshape(1,p))
            for k in range(1,P+1):
                X[j]=np.append(X[j],scaled_train[i-(12*k)])
            y.append(scaled_train[i])
            j+=1
        X=np.array(X).reshape(j,n_input)
        y=np.array(y).reshape(j,1)
        model = Sequential()
        model.add(Dense(4, activation='relu', input_dim=n_input))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X,y,epochs=300)
        forecast = []
        eval_batch = scaled_train[-p:].reshape(1,p)
        u=list(scaled_train[-P*12:])
        for k in range(1,P+1):
            eval_batch=np.append(eval_batch,u[-k*12])
        
        for i in range(len(test)):
            eval_batch=eval_batch.reshape(1,n_input)
#             print(eval_batch)
            current_pred = model.predict(eval_batch)
            forecast.append(current_pred[0])
            u.append(current_pred[0][0])
            if P>0:
                eval_batch=eval_batch[:,:-P]
            eval_batch=np.append(eval_batch[:,1:],current_pred[0][0])
            for k in range(1,P+1):
                eval_batch=np.append(eval_batch,u[-k*12])
        forecast= scaler.inverse_transform(forecast)
        mape+=mean_absolute_percentage_error(test,forecast)
        t+=1
    details['MAPE']=mape/t
    return details   

def prophet(df1,horizon):
    df=df1.reset_index()
    col=df.columns[0]
    df=df.rename(columns={df.columns[0]: "ds", df.columns[1]: "y"})
    df['ds'] = pd.to_datetime(df['ds'])
    n_train = int(len(df)*0.7)
    n_records = len(df)
    MAPE_add=0
    MAPE_mul=0
    j=0
    for l in range(n_train, n_records-horizon+1):
        train, test = df.iloc[0:l], df.iloc[l:l+horizon]
        m = Prophet()
        m.fit(train)
        future = m.make_future_dataframe(periods=len(test),freq='MS')
        forecast = m.predict(future)
        test_predictions = forecast.iloc[-len(test):]['yhat']
        MAPE_add+=mean_absolute_percentage_error(test['y'],test_predictions)

        m = Prophet(seasonality_mode='multiplicative')
        m.fit(train)
        future = m.make_future_dataframe(periods=len(test),freq='MS')
        forecast = m.predict(future)
        test_predictions = forecast.iloc[-len(test):]['yhat']
        MAPE_mul+=mean_absolute_percentage_error(test['y'],test_predictions)
        j+=1
    if (MAPE_add<MAPE_mul or math.isnan(MAPE_mul)):
        return {'mode':'add','MAPE':MAPE_add/j}
    return {'mode':'mul','MAPE':MAPE_mul/j}       

#any model can be used to calculate the accuracy and details for the data frame
df=transform_series(df)
df=outlier(df)
#use the model function now