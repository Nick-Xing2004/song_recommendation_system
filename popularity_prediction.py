import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model():
    df_song=pd.read_csv('./song_recommendation_system/data.csv')
    df_song=df_song.drop(columns=['name', 'release_date', 'explicit', 'mode', 'id', 'key', 'artists'])
    y=df_song['popularity']
    X=df_song.drop(columns=['popularity'])
    X=X.to_numpy()
    y=y.to_numpy()
    popularity_model=LinearRegression()
    popularity_model.fit(X, y)
    return popularity_model

def predict_popularity(parameter_list, model):
    return model.predict(parameter_list)
    
if __name__ == "__main__":
    popularity_model = train_model()
    f1=float(input("please enter feature 1:"))
    f2=float(input("please enter feature 2:"))
    f3=float(input("please enter feature 3:"))
    f4=float(input("please enter feature 4:"))
    f5=float(input("please enter feature 5:"))
    f6=float(input("please enter feature 6:"))
    f7=float(input("please enter feature 7:"))
    f8=float(input("please enter feature 8:"))
    f9=float(input("please enter feature 9:"))
    f10=float(input("please enter feature 10:"))
    f11=float(input("please enter feature 11:"))
    param_lst=[]
    param_lst.append([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11])
    result=predict_popularity(param_lst, popularity_model)
    print(f"the predicted popularity is {result[0]}")