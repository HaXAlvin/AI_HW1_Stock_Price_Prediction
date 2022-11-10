#### 執行環境(詳見requirements.txt)：

* python=3.9.12
* pandas=1.5.0
* numpy=1.23.3
* torch=1.12.1
* matplotlib(optional)=3.6.0

#### 執行過程之輸入與輸出：

同main.py檔案目錄下，需有./Dataset/test.csv與./Dataset/train.csv檔案作為輸入，

在執行完main.py後會產生一./submit.csv檔案。

#### 程式說明：

Line 14~24 為模型物件，有兩Linear層與一Relu層，需傳入input_size與hidden_size。

Line 26 的 func 分別是用來取得資料並作正規劃，type參數可以調整取得，所有資料、8成訓練資料、2成validate資料。

Line 41 ~72 為訓練部分，分為四種Learning Rate來實驗各自訓練300 Epoch，並在 Line 65後對各Learning Rate的Loss-Epoch做圖比較。有一現象為較高的LR的最終Loss通常會較低，但若LR調太高又會使Loss徒增。

Line 74~84 為Validate的部分，將會對Training時的四個Model計算Loss，最終會選擇Loss最低的最為之後Prediction的Model。

Line 87~97 為Prediction的部分，將資料每四筆切割做預測，並產生一submit.csv檔案。
