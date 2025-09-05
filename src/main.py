import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import kagglehub
from kagglehub import kaggle_cache_resolver,exceptions
import numpy as np
import os
import time

def _install_kgh_dataset(filename:str):
    try:
        path = kagglehub.dataset_download(filename)
    except exceptions.HTTPStatus.FAILED_DEPENDENCY as kgherror:
        return kgherror
    except Exception as error:
        return error
    



def data_reader(filename:str, col:str):
    try:
        df = pd.read_csv(filename)
        if col is None:
            return df
        else:
            data =[]
            coldata = df[f"{col}"]
            for obj in range(len(coldata)):
                if coldata[obj] == "NaN":
                    pass
                else:
                    data.append(coldata[obj])
            result = [obj for obj in data ]
            return result
    except Exception as error:
        return error
    
def normalize(filename:str,output:str):
    try:
        '''
        Chuẩn hóa dữ liệu để xử lí 
        quá trình học máy , tránh cho 
        giá trị bị biến động thành giá 
        trị ngoại lai
        Chuẩn hóa theo mẫu Z-score
        '''
        
        df = pd.read_csv(filename)

        
        replaced_pos = r'>|,|sqft|sqm|sqyrd|acre|ground|cent|bigha|marla|kanal|biswa2|aankadam|hectare'
        
        # danh sách các biển độc lập : input
        df["Carpet Area"] = df["Carpet Area"].astype(str).str.replace(replaced_pos,'',regex=True).astype(float)
        df["Super Area"] = df["Super Area"].astype(str).str.replace(replaced_pos,'',regex=True).astype(float)
        df["Bathroom"] = df["Bathroom"].astype(str).str.replace(replaced_pos,'',regex=True).astype(float)
        df["Balcony"] = df["Balcony"].astype(str).str.replace(replaced_pos,'',regex=True).astype(float)
        
       
         #lọc các giữ liệu rỗng , giá trị ngoại lai
        df = df.fillna(df.mean(numeric_only=True))
        ft = df[["Carpet Area","Super Area","Bathroom","Balcony"]]
        lbl = df[['Price (in rupees)']]

        print(ft.values)
        
        # biển đối, chuẩn hóa giá trị
        scaler = StandardScaler()
        lbl_scaler = StandardScaler()

        scaled_ft = scaler.fit_transform(ft)
        scaled_lbl = lbl_scaler.fit_transform(lbl)


        #đưa dữ liệu đã được làm sạch vào một dataset mới 
        normalized_df = pd.DataFrame(scaled_ft, columns=["Carpet Area","Super Area","Bathroom","Balcony"])
        normalized_df["Price (in rupees)"] = scaled_lbl
        normalized_df.to_csv(f"{output}", index=False)
        return "Normalized and formatted"
    except FileNotFoundError as fnf:
        return fnf
    except Exception as error:
        return error
    
def scale_group(filename:str):
    '''
    Gom và chuẩn hóa dữ liệu theo công thức
    theo vector 2 chiều
    '''
    df = pd.read_csv(filename)
    #input
    carpetarea = df["Carpet Area"].astype(float).values
    bathroom = df["Bathroom"].astype(int).values
    balcony = df["Balcony"].astype(int).values
    sparea = df["Super Area"].astype(float).values
    #output
    price = df["Price (in rupees)"].astype(float).values

    # gom dữ liệu thành vector
    x_vector_data = np.vstack([carpetarea,sparea,bathroom,balcony]).T
    # chuyển đổi hàng vector về 1 cột
    output_price = price.reshape(-1, 1)
    # ( x - Mean(X)) / sigma (X)
    input_data = (x_vector_data - np.mean(x_vector_data, axis=0)) / np.std(x_vector_data, axis=0)

    return input_data, output_price

# sai số bình phương
def Means_square_errors(x,y,coffencient, bias):
    '''
    Sử dụng 
    công thức trung bình cộng bình phương sai số
    theo dạng tổng thể

    '''
    prediction = np.dot(x,coffencient)+bias
    mse = np.mean((y-prediction)**2)
    return mse

# cập nhật giá trị
def update_value(x,y,coffencient,bias,learning):
    '''
    Đạo hàm 2 giá trị hệ số chặn, hệ số 
    cập nhật 2 giá trị đó liên tục
    '''
    predictions = np.dot(x,coffencient)+bias
    db = np.mean(predictions-y)
    dw = np.dot(x.T, (predictions-y)) / len(y)

    #điều chỉnh 
    coffencient -= learning*dw
    bias -= learning*db
    return coffencient,bias

def total_bias(filename:str):
    '''Sử dụng bài toán độ lệch
    tổng thể để áp dụng và mô hình huấn
    luyện '''
    try:
        df = pd.read_csv(filename)
        x,y = scale_group(filename)
        data = x.shape[1]
        means = np.mean(x)
        mse = np.mean(np.power((data-means),2))
        return np.sqrt(mse/len(y))
    except FileNotFoundError as fnf:
        return f"fnf: {fnf}"
    except Exception as error:
        return f"internal: {error}"


# huấn luyện mô hình
def training(learning,filename,epochs:int):
    '''
    huấn luyện mô hình cho ra các kết quả 
    mới từ tốc độ học và số lần huấn luyện
    '''
    count = 0
    x,y = scale_group(filename)
    ft = x.shape[1] # chọn kích thước ma trận
    coffencient = np.random.randn(ft,1)
    bias = total_bias(filename)
    period_per_epoch = []
    # bắt đầu huấn luyện
    for obj in range(epochs):
        count +=1
        coffencient,bias = update_value(x,y,coffencient,bias,learning)
        loss = Means_square_errors(x,y,coffencient,bias)
        if obj%100 == 0:
            print(

                f"period:{count}\nLoss: {loss}\nCoffencient {coffencient}\nBias: {bias}"
            )
            period_per_epoch.append(loss)

    print("RESULT")
    print("Loss: ",round(loss,4))


def predict(x,y,filename:str,loss):
    '''
    Dự đoán kết quả giá nhà sau khi 
    tìm được giá trị thích hợp
    '''
    try:
        predictions = np.array

    except FileNotFoundError as fnf:
        return f"fnf: {fnf}"
    except Exception as error:
        return error

if __name__ == "__main__":
    LN = 0.0001
    EPOCHS = 50000
    

    path = "juhibhojani/house-price"
    dir_or = os.path.dirname("D:\\MCT\\ARM_kaggle\\")


    original_file = 'data/house_prices.csv'
    original_filename = os.path.join(dir_or,original_file)
    
    normalize_file = "data/Cleaned_price.csv"
    normalize_filename = os.path.join(dir_or,normalize_file)

    time.sleep(5)
    print("Normalizing...")
    print(normalize(original_filename,normalize_filename))
    time.sleep(5)
    
    records = training(LN,normalize_filename,EPOCHS)
    print(records)


    
    
    # print(data_reader(filename,"Price (in rupees)"))

